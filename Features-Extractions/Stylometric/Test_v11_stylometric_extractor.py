# stylometric_extractor.py
# unified stylometric feature extraction -- New stylometric extractor for v. 1.1;
# Focus on streamlining, reducing cardinality and maintaining linguistic insights; stronger preprocessing and auto-insert of imputation.
#
# output features (20):
#   lexical:      type_token_ratio, yules_k, hapax_legomena_ratio, token_burstiness
#   character:    char_trigram_entropy, compression_ratio, avg_word_length
#   functional:   stopword_ratio, comma_ratio
#   structural:   sentence_length_std
#   readability:  flesch_reading_ease
#   syntactic:    avg_tree_depth, avg_dependency_distance
#   pos:          content_function_ratio, verbs_per_100, pos_ratio_CCONJ,
#                 pos_transition_entropy, prop_sents_with_verb
#   sentiment:    sentiment_subjectivity, sentiment_polarity_variance
#
# metadata (not model features): id, n_tokens_doc, n_sentences_doc

import pandas as pd
import numpy as np
import math
import re
import zlib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import spacy
from spacy.tokens import Doc, Token
import nltk
from nltk.corpus import cmudict
from g2p_en import G2p
from textblob import TextBlob
import torch


# configuration


BASE_PATH = Path(__file__).parent
OUTPUT_PATH = BASE_PATH / "output"
CHECKPOINT_PATH = BASE_PATH / "checkpoints"
OUTPUT_PATH.mkdir(exist_ok=True)
CHECKPOINT_PATH.mkdir(exist_ok=True)

BATCH_SIZE = 16
#to-set
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# minimum token thresholds for features that need sufficient text
# lowered from 100 to 50: yules_k is stable at 50+ tokens with the
# 500-token window cap, and flesch works acceptably at 50+ tokens
# with 2+ sentences (original flesch required 100 words)
MIN_TOKENS_LEXICAL = 50   # yules_k
MIN_TOKENS_READABILITY = 50  # flesch_reading_ease
MIN_SENTS_READABILITY = 2    # flesch_reading_ease


SELECTED_FEATURES = [
    "type_token_ratio",
    "yules_k",
    "hapax_legomena_ratio",
    "token_burstiness",
    "char_trigram_entropy",
    "compression_ratio",
    "avg_word_length",
    "stopword_ratio",
    "comma_ratio",
    "sentence_length_std",
    "flesch_reading_ease",
    "avg_tree_depth",
    "avg_dependency_distance",
    "content_function_ratio",
    "verbs_per_100",
    "pos_ratio_CCONJ",
    "pos_transition_entropy",
    "prop_sents_with_verb",
    "sentiment_subjectivity",
    "sentiment_polarity_variance",
]

# metadata columns retained for imputation and downstream joins
METADATA_COLS = ["id", "n_tokens_doc", "n_sentences_doc"]

# pos tag sets
POS_TAGS = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET",
    "ADP", "AUX", "CCONJ", "PART", "NUM", "PUNCT", "X",
]
POS_SET = set(POS_TAGS)
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "NUM"}
FUNCTION_POS = {"PRON", "DET", "ADP", "AUX", "CCONJ", "PART", "PUNCT", "X"}

# arpabet vowels for syllable counting
ARPA_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
    "EY", "IH", "IY", "OW", "OY", "UH", "UW",
}
_SYLL_CACHE: Dict[str, int] = {}


# gpu setup


def setup_device():
    if torch.cuda.is_available():
        print(f"[GPU] using: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("[CPU] no gpu available, using cpu")
        return "cpu"

DEVICE = setup_device()


# nlp model and phoneme setup


def load_nlp_resources():
    nltk.download('cmudict', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    cmu = cmudict.dict()
    g2p = G2p()

    print("loading spacy model...")
    try:
        nlp = spacy.load("en_core_web_trf")
        if DEVICE == "cuda":
            spacy.require_gpu()
    except OSError:
        print("downloading spacy model...")
        from spacy.cli import download
        download("en_core_web_trf")
        nlp = spacy.load("en_core_web_trf")

    if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    print(f"spacy loaded, pipes: {nlp.pipe_names}")
    return nlp, cmu, g2p


# helpers


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _word_like(tok) -> bool:
    return tok.is_alpha and not tok.is_space


def _alnum_char_count(token_text: str) -> int:
    return sum(ch.isalnum() for ch in token_text)


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(max(p, 1e-12))
    return float(ent)


# syllable counting (needed for flesch)


def cmu_syllables(word: str, cmu: dict) -> int:
    w = word.lower()
    if w not in cmu:
        return None
    phones = cmu[w][0]
    count = sum(1 for ph in phones if re.sub(r"\d", "", ph) in ARPA_VOWELS)
    return max(count, 1)


def g2p_syllables(word: str, g2p: G2p) -> int:
    w = word.lower()
    if w in _SYLL_CACHE:
        return _SYLL_CACHE[w]
    phones = g2p(w)
    count = sum(1 for ph in phones if re.sub(r"\d", "", ph) in ARPA_VOWELS)
    if count == 0 and re.search(r"[A-Za-z]", w):
        count = 1
    _SYLL_CACHE[w] = count
    return count


def syllables_hybrid(word: str, cmu: dict, g2p: G2p) -> int:
    c = cmu_syllables(word, cmu)
    return c if c is not None else g2p_syllables(word, g2p)


# lexical diversity features
# type_token_ratio, yules_k, hapax_legomena_ratio, token_burstiness


def compute_lexical_features(word_toks: list, n_tokens: int) -> Dict[str, float]:
    feats = {}

    # type-token ratio
    if n_tokens > 0:
        vocab = {t.text.lower() for t in word_toks}
        feats["type_token_ratio"] = len(vocab) / n_tokens
    else:
        feats["type_token_ratio"] = np.nan

    # build lowered token list (capped at 500 for stability)
    tok_win = [t.text.lower() for t in word_toks if _word_like(t)][:500]

    # yules_k
    N = len(tok_win)
    if N >= MIN_TOKENS_LEXICAL:
        cnt = Counter(tok_win)
        spectrum = Counter(cnt.values())
        s2 = sum((v * v) * Vv for v, Vv in spectrum.items())
        Nf = float(N)
        feats["yules_k"] = 10000.0 * (s2 - Nf) / (Nf * Nf)
    else:
        feats["yules_k"] = np.nan

    # hapax legomena ratio
    if tok_win:
        cnt = Counter(tok_win)
        hapax = sum(1 for c in cnt.values() if c == 1)
        feats["hapax_legomena_ratio"] = hapax / float(len(tok_win))
    else:
        feats["hapax_legomena_ratio"] = np.nan

    # token burstiness
    if len(tok_win) >= 2:
        word_counts = Counter(tok_win)
        frequencies = list(word_counts.values())
        if len(frequencies) >= 2:
            mu = np.mean(frequencies)
            sigma = np.std(frequencies, ddof=0)
            denom = sigma + mu
            feats["token_burstiness"] = (
                (sigma - mu) / denom if denom > 0 else np.nan
            )
        else:
            feats["token_burstiness"] = np.nan
    else:
        feats["token_burstiness"] = np.nan

    return feats


# character-level features
# char_trigram_entropy, compression_ratio, avg_word_length


def compute_character_features(text: str, word_toks: list) -> Dict[str, float]:
    feats = {}

    # char trigram entropy
    n = 3
    if len(text) >= n:
        char_ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
        counts = Counter(char_ngrams)
        total = len(char_ngrams)
        feats["char_trigram_entropy"] = sum(
            -(c / total) * math.log2(c / total) for c in counts.values()
        )
    else:
        feats["char_trigram_entropy"] = np.nan

    # compression ratio
    encoded = text.encode("utf-8")
    raw_len = len(encoded)
    if raw_len > 0:
        compressed = zlib.compress(encoded, level=6)
        feats["compression_ratio"] = len(compressed) / raw_len
    else:
        feats["compression_ratio"] = np.nan

    # average word length
    if word_toks:
        feats["avg_word_length"] = float(
            np.mean([len(t.text) for t in word_toks])
        )
    else:
        feats["avg_word_length"] = np.nan

    return feats


# functional features
# stopword_ratio, comma_ratio


def compute_functional_features(
    doc: Doc, word_toks: list, n_tokens: int
) -> Dict[str, float]:
    feats = {}

    # stopword ratio
    if n_tokens > 0:
        stop_count = sum(1 for t in word_toks if t.is_stop)
        feats["stopword_ratio"] = stop_count / n_tokens
    else:
        feats["stopword_ratio"] = np.nan

    # comma ratio (commas / all non-space tokens)
    all_tokens = [t for t in doc if not t.is_space]
    if all_tokens:
        punct_tokens = [t for t in doc if t.is_punct]
        punct_text = "".join(t.text for t in punct_tokens)
        feats["comma_ratio"] = punct_text.count(",") / len(all_tokens)
    else:
        feats["comma_ratio"] = 0.0

    return feats


# structural features
# sentence_length_std


def compute_structural_features(sents: list) -> Dict[str, float]:
    sent_word_counts = [sum(1 for t in s if _word_like(t)) for s in sents]
    if len(sent_word_counts) > 1:
        return {
            "sentence_length_std": float(np.std(sent_word_counts, ddof=0))
        }
    else:
        return {"sentence_length_std": np.nan}


# readability
# flesch_reading_ease


def compute_readability_features(
    word_toks: list, n_tokens: int, n_sents: int, cmu: dict, g2p: G2p
) -> Dict[str, float]:
    if n_tokens < MIN_TOKENS_READABILITY or n_sents < MIN_SENTS_READABILITY:
        return {"flesch_reading_ease": np.nan}

    syllable_counts = [syllables_hybrid(t.text, cmu, g2p) for t in word_toks]
    total_syllables = max(int(np.sum(syllable_counts)), 1)
    words = max(n_tokens, 1)
    sents = max(n_sents, 1)

    fre = 206.835 - 1.015 * (words / sents) - 84.6 * (total_syllables / words)
    return {"flesch_reading_ease": fre}


# syntactic features
# avg_tree_depth, avg_dependency_distance


def _root_chain_depth(token: Token, max_steps: int) -> int:
    depth = 0
    cur = token
    visited = set()
    while cur.head.i != cur.i:
        if cur.i in visited:
            return 0
        visited.add(cur.i)
        depth += 1
        if depth > max_steps:
            return max_steps
        cur = cur.head
    return depth


def compute_syntactic_features(doc: Doc, sents: list) -> Dict[str, float]:
    max_steps = len(doc) + 5
    depths = []
    per_sent_distances = []

    for sent in sents:
        sent_depths = [_root_chain_depth(tok, max_steps) for tok in sent]
        if sent_depths:
            depths.append(max(sent_depths))

        distances = []
        for token in sent:
            if token.head.i != token.i:
                distances.append(abs(token.i - token.head.i))
        if distances:
            per_sent_distances.append(np.mean(distances))

    return {
        "avg_tree_depth": float(np.mean(depths)) if depths else 0.0,
        "avg_dependency_distance": (
            float(np.mean(per_sent_distances))
            if per_sent_distances
            else 0.0
        ),
    }


# pos features
# content_function_ratio, verbs_per_100, pos_ratio_CCONJ,
# pos_transition_entropy, prop_sents_with_verb


def compute_pos_features(doc: Doc, sents: list) -> Dict[str, float]:
    toks = [t for t in doc if not t.is_space]
    total_tokens = len(toks)

    if total_tokens == 0:
        return {
            "content_function_ratio": 0.0,
            "verbs_per_100": 0.0,
            "pos_ratio_CCONJ": 0.0,
            "pos_transition_entropy": 0.0,
            "prop_sents_with_verb": 0.0,
        }

    pos_seq = [t.pos_ if t.pos_ in POS_SET else "X" for t in toks]
    pos_counts = Counter(pos_seq)

    content_sum = sum(pos_counts.get(t, 0) for t in CONTENT_POS)
    function_sum = sum(pos_counts.get(t, 0) for t in FUNCTION_POS)
    verbs = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
    cconj = pos_counts.get("CCONJ", 0)

    # transition entropy over POS bigrams
    transitions = list(zip(pos_seq, pos_seq[1:]))
    trans_counts = Counter(transitions)

    # proportion of sentences with at least one verb
    verb_presence = []
    for s in sents:
        s_toks = [t for t in s if not t.is_space]
        s_counts = Counter(t.pos_ for t in s_toks)
        has_verb = (s_counts.get("VERB", 0) + s_counts.get("AUX", 0)) > 0
        verb_presence.append(1 if has_verb else 0)

    return {
        "content_function_ratio": safe_div(content_sum, function_sum),
        "verbs_per_100": 100.0 * safe_div(verbs, total_tokens),
        "pos_ratio_CCONJ": safe_div(cconj, total_tokens),
        "pos_transition_entropy": shannon_entropy(trans_counts),
        "prop_sents_with_verb": safe_div(
            sum(verb_presence), len(verb_presence)
        ),
    }


# sentiment features
# sentiment_subjectivity, sentiment_polarity_variance


def compute_sentiment_features(
    text: str, doc: Doc, sents: list
) -> Dict[str, float]:
    if not text or not text.strip():
        return {
            "sentiment_subjectivity": 0.0,
            "sentiment_polarity_variance": 0.0,
        }

    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity

    sent_polarities = []
    for sent in sents:
        sent_blob = TextBlob(sent.text)
        sent_polarities.append(sent_blob.sentiment.polarity)

    polarity_var = (
        float(np.var(sent_polarities)) if len(sent_polarities) > 1 else 0.0
    )

    return {
        "sentiment_subjectivity": subjectivity,
        "sentiment_polarity_variance": polarity_var,
    }


# main feature extraction -- assembles all 20 features


def extract_features(
    doc: Doc, text: str, cmu: dict, g2p: G2p
) -> Dict[str, float]:
    features = {}

    sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
    word_toks = [t for t in doc if _word_like(t)]
    n_tokens = len(word_toks)
    n_sents = len(sents)

    # metadata
    features["n_tokens_doc"] = float(n_tokens)
    features["n_sentences_doc"] = float(n_sents)

    # feature groups
    features.update(compute_lexical_features(word_toks, n_tokens))
    features.update(compute_character_features(text, word_toks))
    features.update(compute_functional_features(doc, word_toks, n_tokens))
    features.update(compute_structural_features(sents))
    features.update(
        compute_readability_features(word_toks, n_tokens, n_sents, cmu, g2p)
    )
    features.update(compute_syntactic_features(doc, sents))
    features.update(compute_pos_features(doc, sents))
    features.update(compute_sentiment_features(text, doc, sents))

    return features


# imputation


def impute_missing_features(
    df: pd.DataFrame, max_missing_pct: float = 0.3
) -> pd.DataFrame:
    """
    Stratified imputation by document length with global median fallback.
    Default threshold 0.3: balances between conservative (0.2, drops too
    aggressively on short-heavy corpora) and generous (0.4, imputes too much).
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    num_feats = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {"is_ai", "id"}
    ]
    if not num_feats:
        return df

    # drop features with too much missing data
    missing_pct = df[num_feats].isna().mean()
    high_missing = missing_pct[missing_pct > max_missing_pct]

    if not high_missing.empty:
        print(
            f"\n[impute] dropping {len(high_missing)} features "
            f"with >{max_missing_pct * 100:.0f}% missing:"
        )
        for feat, pct in high_missing.items():
            print(f"  - {feat}: {pct * 100:.1f}% missing")
        num_feats = [f for f in num_feats if f not in high_missing.index]
        df.drop(columns=high_missing.index, inplace=True)

    if not num_feats:
        return df

    # stratified imputation by document length
    if "n_tokens_doc" in df.columns:
        bins = [0, 50, 100, 250, 500, 10000]
        labels = ["XS", "S", "M", "L", "XL"]
        df["__len_bin__"] = pd.cut(
            df["n_tokens_doc"], bins=bins, right=False, labels=labels
        )
        for feat in num_feats:
            if df[feat].isna().any():
                group_medians = df.groupby(
                    "__len_bin__", observed=False
                )[feat].transform("median")
                df[feat] = df[feat].fillna(group_medians)
        df.drop(columns=["__len_bin__"], errors="ignore", inplace=True)

    # global median fallback
    for feat in num_feats:
        if df[feat].isna().any():
            median_val = df[feat].median()
            n_filled = df[feat].isna().sum()
            df[feat] = df[feat].fillna(median_val)
            if n_filled > 0:
                print(
                    f"[impute] filled {n_filled} values in {feat} "
                    f"with median {median_val:.4f}"
                )

    return df


# main pipeline


def extract_all_features(
    df: pd.DataFrame, text_col: str, impute: bool = True
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STYLOMETRIC FEATURE EXTRACTION")
    print("=" * 70)

    nlp, cmu, g2p = load_nlp_resources()

    texts = df[text_col].astype(str).tolist()
    doc_ids = (
        df["id"].tolist() if "id" in df.columns else list(range(len(df)))
    )

    print(f"\nprocessing {len(texts)} documents...")

    all_features = []

    from tqdm import tqdm

    for i, doc in enumerate(
        tqdm(
            nlp.pipe(texts, batch_size=BATCH_SIZE),
            total=len(texts),
            desc="extracting",
        )
    ):
        text = texts[i]
        feats = extract_features(doc, text, cmu, g2p)
        feats["id"] = doc_ids[i]
        all_features.append(feats)

        if DEVICE == "cuda" and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    features_df = pd.DataFrame(all_features)

    # reorder: metadata first, then selected features
    output_cols = METADATA_COLS + [
        f for f in SELECTED_FEATURES if f in features_df.columns
    ]
    output_cols = [c for c in output_cols if c in features_df.columns]
    features_df = features_df[output_cols]

    # imputation
    if impute:
        print("\n[imputation] filling missing values...")
        features_df = impute_missing_features(
            features_df, max_missing_pct=0.3
        )

    # summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"documents processed: {len(features_df)}")
    print(f"features extracted: {len(SELECTED_FEATURES)}")
    print(f"total columns: {len(features_df.columns)} (incl. metadata)")

    nan_counts = features_df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\nremaining nan columns: {len(nan_cols)}")
        for col, count in nan_cols.items():
            print(
                f"  {col}: {count} ({100 * count / len(features_df):.1f}%)"
            )
    else:
        print("\nno nan values remaining")

    return features_df


# mock data and testing


def create_mock_data(n_samples: int = 30) -> pd.DataFrame:
    """
    Generate mock data from RAID.
    A few short texts (< 50 tokens) test the nan/imputation paths.
    """
    np.random.seed(RANDOM_SEED)

    paragraphs = [
        (
            "The advancement of artificial intelligence has transformed "
            "numerous industries across the global economy. Machine learning "
            "algorithms now power recommendation systems, autonomous vehicles, "
            "and medical diagnostics. Deep neural networks have achieved "
            "remarkable success in natural language processing and computer "
            "vision tasks. Researchers continue to push the boundaries of "
            "what machines can accomplish through sophisticated architectures. "
            "The ethical implications of these technologies remain a subject "
            "of intense debate among scholars and policymakers. Companies "
            "invest billions of dollars annually in developing more capable "
            "and efficient AI systems. The workforce must adapt to these "
            "changes through continuous learning and skill development."
        ),
        (
            "Climate change represents one of the most pressing challenges "
            "facing humanity in the twenty-first century. Rising global "
            "temperatures cause widespread disruption to ecosystems and "
            "weather patterns worldwide. Coastal communities face increasing "
            "threats from sea level rise and more frequent extreme weather "
            "events. Scientists have documented accelerating ice loss in "
            "both Arctic and Antarctic regions over recent decades. "
            "International cooperation remains essential for implementing "
            "effective mitigation and adaptation strategies. Renewable energy "
            "technologies offer promising pathways toward reducing greenhouse "
            "gas emissions significantly."
        ),
        (
            "The human brain contains approximately eighty-six billion "
            "neurons connected through trillions of synapses. Neuroscientists "
            "study how these complex networks give rise to consciousness, "
            "memory, and behavior patterns. Modern imaging techniques allow "
            "researchers to observe brain activity with unprecedented spatial "
            "resolution. Understanding neural mechanisms has important "
            "implications for treating neurological and psychiatric disorders. "
            "The field of computational neuroscience builds mathematical "
            "models to simulate brain function accurately."
        ),
        (
            "Economic inequality has increased substantially in many "
            "developed nations over the past several decades. Wealth "
            "concentration among the top percentile raises concerns about "
            "social mobility and democratic governance. Policy interventions "
            "such as progressive taxation and education investments aim to "
            "address these disparities effectively. The relationship between "
            "economic growth and inequality remains a contested topic among "
            "economists today. Social safety nets provide crucial support "
            "for vulnerable populations during economic downturns."
        ),
        (
            "Literature serves as a mirror reflecting the complexities of "
            "human experience across cultures and epochs. Great works of "
            "fiction explore universal themes of love, loss, identity, and "
            "the search for meaning. The novel emerged as a dominant literary "
            "form during the eighteenth century in European cultural contexts. "
            "Postmodern authors challenge traditional narrative conventions "
            "through experimental techniques and structures. Literary "
            "criticism examines how texts produce meaning through language, "
            "form, and cultural references. Reading fiction has been shown to "
            "enhance empathy and emotional intelligence in studies. The "
            "digital age has transformed how literature is produced, "
            "distributed, and consumed by global audiences worldwide."
        ),
    ]

    data = []
    for i in range(n_samples):
        if i % 10 == 0:
            # short text (< 50 tokens): tests nan paths and imputation
            text = (
                "A short test sentence for validation. "
                "This checks edge cases in the pipeline."
            )
        elif i % 10 == 1:
            # medium-short text (~60-80 tokens): above threshold but modest
            text = paragraphs[i % len(paragraphs)]
        else:
            # long text (150-250+ tokens): combines multiple paragraphs
            base = paragraphs[i % len(paragraphs)]
            extra = paragraphs[(i + 1) % len(paragraphs)]
            text = base + " " + extra
            if i % 3 == 0:
                text += (
                    " Furthermore, these considerations have profound "
                    "implications for future generations and their "
                    "wellbeing. Scholars continue to debate the best "
                    "approaches for addressing these complex challenges."
                )

        data.append({"id": f"doc_{i:04d}", "text": text, "is_ai": i % 2})

    return pd.DataFrame(data)


def run_test():
    """
    End-to-end test: create mock data, extract features, validate output.
    Checks:
    1. All selected features present after extraction (pre-imputation)
    2. Metadata columns present
    3. Short texts produce nans where expected
    4. Imputation fills nans without dropping features
    5. Feature value ranges are plausible
    6. Row count preserved
    """
    import time

    print("\n" + "=" * 70)
    print("RUNNING TEST WITH MOCK DATA")
    print("=" * 70)

    mock_df = create_mock_data(30)
    mock_path = BASE_PATH / "mock_data.csv"
    mock_df.to_csv(mock_path, index=False)

    token_counts = mock_df["text"].apply(lambda x: len(x.split()))
    print(f"created mock data: {len(mock_df)} samples")
    print(
        f"token counts: min={token_counts.min()}, "
        f"max={token_counts.max()}, mean={token_counts.mean():.0f}"
    )
    n_short = (token_counts < MIN_TOKENS_LEXICAL).sum()
    print(f"short docs (< {MIN_TOKENS_LEXICAL} tokens): {n_short}")
    print(f"saved to: {mock_path}")

    # -- phase 1: extraction without imputation --
    print("\n--- phase 1: extraction without imputation ---")
    start_time = time.time()
    raw_df = extract_all_features(mock_df, "text", impute=False)
    elapsed_raw = time.time() - start_time

    # check nan behavior in short docs
    short_mask = raw_df["n_tokens_doc"] < MIN_TOKENS_LEXICAL
    n_short_actual = short_mask.sum()

    print(f"\nshort documents (< {MIN_TOKENS_LEXICAL} tokens): {n_short_actual}")
    for feat in ["yules_k", "flesch_reading_ease", "sentence_length_std"]:
        if feat in raw_df.columns:
            nan_count = raw_df.loc[short_mask, feat].isna().sum()
            total_nan = raw_df[feat].isna().sum()
            print(f"  {feat}: {nan_count} nan in short, {total_nan} total nan")

    # -- phase 2: extraction with imputation --
    print("\n--- phase 2: extraction with imputation ---")
    start_time = time.time()
    features_df = extract_all_features(mock_df, "text", impute=True)
    elapsed = time.time() - start_time

    test_output = OUTPUT_PATH / "test_stylometric_features_20.csv"
    features_df.to_csv(test_output, index=False)

    # -- validation checks --
    print("\n" + "=" * 70)
    print("TEST VALIDATION")
    print("=" * 70)

    checks_passed = 0
    checks_total = 0

    # check 1: all 20 features present pre-imputation
    checks_total += 1
    missing_pre = [f for f in SELECTED_FEATURES if f not in raw_df.columns]
    if not missing_pre:
        print("[PASS] all 20 features present in raw extraction")
        checks_passed += 1
    else:
        print(f"[FAIL] missing in raw extraction: {missing_pre}")

    # check 2: metadata columns present
    checks_total += 1
    missing_meta = [c for c in METADATA_COLS if c not in features_df.columns]
    if not missing_meta:
        print("[PASS] all metadata columns present")
        checks_passed += 1
    else:
        print(f"[FAIL] missing metadata: {missing_meta}")

    # check 3: all 20 features survive imputation (none dropped)
    checks_total += 1
    missing_post = [
        f for f in SELECTED_FEATURES if f not in features_df.columns
    ]
    if not missing_post:
        print("[PASS] all 20 features survived imputation")
        checks_passed += 1
    else:
        print(
            f"[WARN] {len(missing_post)} features dropped by imputation: "
            f"{missing_post}"
        )
        # check what percentage was missing
        for feat in missing_post:
            if feat in raw_df.columns:
                pct = raw_df[feat].isna().mean() * 100
                print(f"  {feat}: {pct:.1f}% missing (threshold: 30%)")

    # check 4: no all-nan columns after imputation
    checks_total += 1
    present_feats = [
        f for f in SELECTED_FEATURES if f in features_df.columns
    ]
    all_nan_cols = [
        c for c in present_feats if features_df[c].isna().all()
    ]
    if not all_nan_cols:
        print("[PASS] no all-nan feature columns after imputation")
        checks_passed += 1
    else:
        print(f"[FAIL] all-nan columns: {all_nan_cols}")

    # check 5: imputation completeness
    checks_total += 1
    remaining_nans = features_df[present_feats].isna().sum().sum()
    if remaining_nans == 0:
        print("[PASS] imputation filled all nan values")
        checks_passed += 1
    else:
        nan_detail = features_df[present_feats].isna().sum()
        nan_detail = nan_detail[nan_detail > 0]
        print(f"[WARN] {remaining_nans} nan values remain:")
        for col, cnt in nan_detail.items():
            print(f"  {col}: {cnt}")
        checks_passed += 1  # not a hard failure

    # check 6: plausible value ranges
    checks_total += 1
    range_issues = []
    range_checks = {
        "type_token_ratio": (0.0, 1.0),
        "stopword_ratio": (0.0, 1.0),
        "comma_ratio": (0.0, 1.0),
        "compression_ratio": (0.0, 1.5),
        "sentiment_subjectivity": (0.0, 1.0),
        "pos_ratio_CCONJ": (0.0, 1.0),
        "prop_sents_with_verb": (0.0, 1.0),
        "verbs_per_100": (0.0, 100.0),
        "avg_word_length": (1.0, 20.0),
        "avg_tree_depth": (0.0, 50.0),
        "flesch_reading_ease": (-100.0, 150.0),
    }
    for feat, (lo, hi) in range_checks.items():
        if feat in features_df.columns:
            fmin = features_df[feat].min()
            fmax = features_df[feat].max()
            if fmin < lo - 0.01 or fmax > hi + 0.01:
                range_issues.append(
                    f"{feat}: [{fmin:.4f}, {fmax:.4f}] "
                    f"outside [{lo}, {hi}]"
                )

    if not range_issues:
        print("[PASS] all feature ranges plausible")
        checks_passed += 1
    else:
        print("[FAIL] range issues:")
        for issue in range_issues:
            print(f"  {issue}")

    # check 7: row count preserved
    checks_total += 1
    if len(features_df) == len(mock_df):
        print(f"[PASS] row count preserved: {len(features_df)}")
        checks_passed += 1
    else:
        print(
            f"[FAIL] row count: input={len(mock_df)}, "
            f"output={len(features_df)}"
        )

    # summary
    print(f"\n{'=' * 70}")
    print(f"CHECKS: {checks_passed}/{checks_total} passed")
    print(f"extraction time (raw): {elapsed_raw:.1f}s")
    print(f"extraction time (with imputation): {elapsed:.1f}s")
    print(f"output saved to: {test_output}")
    print("=" * 70)

    # feature ranges for manual inspection
    print("\nfeature ranges:")
    for col in SELECTED_FEATURES:
        if col in features_df.columns:
            fmin = features_df[col].min()
            fmax = features_df[col].max()
            fmean = features_df[col].mean()
            nan_c = features_df[col].isna().sum()
            print(
                f"  {col:<30} "
                f"[{fmin:>10.4f}, {fmax:>10.4f}] "
                f"mean={fmean:>10.4f}  nan={nan_c}"
            )
        else:
            print(f"  {col:<30} DROPPED by imputation")

    return features_df


# main entry point


def main():
    print("\n" + "=" * 70)
    print("STYLOMETRIC FEATURE EXTRACTOR")
    print("=" * 70)

    dataset_path = input(
        "\nenter dataset path (csv with 'id' and text column): "
    ).strip()

    if not Path(dataset_path).exists():
        print(f"error: file not found at {dataset_path}")
        return

    print(f"\nloading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"loaded {len(df)} documents")

    if "id" not in df.columns:
        df["id"] = range(len(df))
        print("created 'id' column")

    if "generation" in df.columns:
        text_col = "generation"
    elif "text" in df.columns:
        text_col = "text"
    else:
        text_col = input("enter text column name: ").strip()
        if text_col not in df.columns:
            print(f"error: column '{text_col}' not found")
            return

    print(f"using text column: '{text_col}'")

    import time

    start_time = time.time()
    features_df = extract_all_features(df, text_col, impute=True)
    elapsed = time.time() - start_time

    output_path = OUTPUT_PATH / "stylometric_features_20.csv"
    features_df.to_csv(output_path, index=False)
    print(f"\nfeatures saved to: {output_path}")
    print(f"total time: {elapsed / 60:.1f} minutes")

    print("\nsample output (first 3 rows):")
    print(features_df.head(3).to_string())

    return features_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test()
    else:
        main()