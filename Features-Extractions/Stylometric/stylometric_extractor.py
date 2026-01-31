# stylometric_extractor.py
# unified stylometric feature extraction

import pandas as pd
import numpy as np
import math
import re
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Set
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
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


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

# arpabet vowels for syllable counting
ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
_SYLL_CACHE: Dict[str, int] = {}

# pos tag configuration
POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX", "CCONJ", "PART", "NUM", "PUNCT", "X"]
POS_SET = set(POS_TAGS)
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "NUM"}
FUNCTION_POS = {"PRON", "DET", "ADP", "AUX", "CCONJ", "PART", "PUNCT", "X"}


# syllable counting utilities


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


# helper functions


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0

def _word_like(tok) -> bool:
    return tok.is_alpha and not tok.is_space

def _alnum_char_count(token_text: str) -> int:
    return sum(ch.isalnum() for ch in token_text)

def extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(max(p, 1e-12))
    return float(ent)


# readability features (handmade), can be interchanged with other libraries (e.g textatistic, readability etc.)


def can_compute_readability(n_tokens: int, n_sents: int) -> bool:
    return (n_tokens >= 100) and (n_sents >= 3)

def compute_readability(n_tokens: int, n_sents: int, syllables: List[int], chars_per_word: float, complex_words: int, polysyllables: int) -> Dict[str, float]:
    out = {"flesch_reading_ease": np.nan, "gunning_fog": np.nan, "smog_index": np.nan, "automated_readability_index": np.nan}
    
    if not can_compute_readability(n_tokens, n_sents):
        return out
    
    words = max(n_tokens, 1)
    sents = max(n_sents, 1)
    total_syllables = max(int(np.sum(syllables)), 1)
    
    out["flesch_reading_ease"] = 206.835 - 1.015 * (words / sents) - 84.6 * (total_syllables / words)
    out["gunning_fog"] = 0.4 * ((words / sents) + 100.0 * (complex_words / words))
    out["smog_index"] = (1.043 * math.sqrt(30.0 * (polysyllables / sents)) + 3.1291) if polysyllables > 0 else np.nan
    out["automated_readability_index"] = 4.71 * chars_per_word + 0.5 * (words / sents) - 21.43
    
    return out


# ngram features, minimum thresholds; can be modifiable


def safe_ngram_stats(tokens: List[str], n: int = 2, min_ngrams: int = 100) -> Dict[str, float]:
    if len(tokens) < n:
        return {"diversity": np.nan, "entropy": np.nan}
    
    grams = extract_ngrams(tokens, n)
    if len(grams) < min_ngrams:
        return {"diversity": np.nan, "entropy": np.nan}
    
    counts = Counter(grams)
    diversity = len(counts) / len(grams)
    probs = np.array(list(counts.values()), dtype=float) / len(grams)
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {"diversity": diversity, "entropy": entropy}


# lexical diversity features


def hapax_legomena_ratio(tokens: List[str]) -> float:
    if not tokens:
        return np.nan
    cnt = Counter(tokens)
    hapax = sum(1 for c in cnt.values() if c == 1)
    return hapax / float(len(tokens))

def hapax_type_ratio(tokens: List[str]) -> float:
    if not tokens:
        return np.nan
    cnt = Counter(tokens)
    types = len(cnt)
    if types == 0:
        return np.nan
    hapax = sum(1 for c in cnt.values() if c == 1)
    return hapax / float(types)

def yules_k(tokens: List[str], min_tokens: int = 100) -> float:
    N = len(tokens)
    if N < min_tokens:
        return np.nan
    cnt = Counter(tokens)
    spectrum = Counter(cnt.values())
    s2 = sum((v * v) * Vv for v, Vv in spectrum.items())
    Nf = float(N)
    return 10000.0 * (s2 - Nf) / (Nf * Nf)

def calculate_burstiness(tokens: List[str]) -> float:
    if len(tokens) < 2:
        return np.nan
    word_counts = Counter(tokens)
    frequencies = list(word_counts.values())
    if len(frequencies) < 2:
        return np.nan
    mu = np.mean(frequencies)
    sigma = np.std(frequencies, ddof=0)
    if mu + sigma == 0:
        return np.nan
    return (sigma - mu) / (sigma + mu)


# character-level features


def character_ngram_features(text: str, n: int = 3) -> Tuple[float, float]:
    if len(text) < n:
        return np.nan, np.nan
    char_ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not char_ngrams:
        return np.nan, np.nan
    diversity = len(set(char_ngrams)) / len(char_ngrams)
    counts = Counter(char_ngrams)
    total = len(char_ngrams)
    entropy = sum(-(c/total) * math.log2(c/total) for c in counts.values())
    return diversity, entropy

def compression_features(text: str) -> Dict[str, float]:
    encoded = text.encode("utf-8")
    raw_len = len(encoded)
    if raw_len == 0:
        return {"compression_ratio": np.nan}
    compressed = zlib.compress(encoded, level=6)
    return {"compression_ratio": len(compressed) / raw_len}

def character_statistics(text: str) -> Dict[str, float]:
    if not text:
        return {"uppercase_ratio": np.nan, "digit_ratio": np.nan, "whitespace_ratio": np.nan, "unique_char_count": np.nan}
    total = len(text)
    return {
        "uppercase_ratio": sum(1 for c in text if c.isupper()) / total,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / total,
        "whitespace_ratio": sum(1 for c in text if c.isspace()) / total,
        "unique_char_count": float(len(set(text)))
    }


# dependency tree features


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

def dependency_tree_features(doc: Doc) -> Dict[str, float]:
    sentences = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
    if not sentences:
        sentences = [doc]
    
    max_steps = len(doc) + 5
    depths = []
    per_sent_distances = []
    left_deps, right_deps = 0, 0
    
    for sent in sentences:
        sent_depths = [_root_chain_depth(tok, max_steps) for tok in sent]
        if sent_depths:
            depths.append(max(sent_depths))
        
        distances = []
        for token in sent:
            if token.head.i != token.i:
                d = abs(token.i - token.head.i)
                distances.append(d)
                if token.i < token.head.i:
                    left_deps += 1
                else:
                    right_deps += 1
        if distances:
            per_sent_distances.append(np.mean(distances))
    
    total_deps = left_deps + right_deps
    return {
        "avg_tree_depth": float(np.mean(depths)) if depths else 0.0,
        "max_tree_depth": float(max(depths)) if depths else 0.0,
        "avg_dependency_distance": float(np.mean(per_sent_distances)) if per_sent_distances else 0.0,
        "left_dependency_ratio": left_deps / total_deps if total_deps else 0.0
    }


# punctuation features


def punctuation_patterns(doc: Doc) -> Dict[str, float]:
    all_tokens = [t for t in doc if not t.is_space]
    punct_tokens = [t for t in doc if t.is_punct]
    
    if not all_tokens:
        return {k: 0.0 for k in ["comma_ratio", "period_ratio", "question_ratio", "exclamation_ratio", "semicolon_ratio", "colon_ratio", "quote_ratio"]}
    
    total = len(all_tokens)
    punct_text = ''.join([t.text for t in punct_tokens])
    
    return {
        "comma_ratio": punct_text.count(',') / total,
        "period_ratio": punct_text.count('.') / total,
        "question_ratio": punct_text.count('?') / total,
        "exclamation_ratio": punct_text.count('!') / total,
        "semicolon_ratio": punct_text.count(';') / total,
        "colon_ratio": punct_text.count(':') / total,
        "quote_ratio": (punct_text.count('"') + punct_text.count("'")) / total
    }


# sentiment features


def sentiment_features(text: str, doc: Doc) -> Dict[str, float]:
    if not text or not text.strip():
        return {
            "sentiment_polarity": 0.0, "sentiment_subjectivity": 0.0,
            "sentiment_polarity_variance": 0.0, "neutral_sentence_ratio": 0.0,
            "positive_word_ratio": 0.0, "negative_word_ratio": 0.0
        }
    
    blob = TextBlob(text)
    features = {
        "sentiment_polarity": blob.sentiment.polarity,
        "sentiment_subjectivity": blob.sentiment.subjectivity
    }
    
    sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
    sent_polarities = []
    neutral_count = 0
    
    for sent in sents:
        sent_blob = TextBlob(sent.text)
        polarity = sent_blob.sentiment.polarity
        sent_polarities.append(polarity)
        if abs(polarity) < 0.1:
            neutral_count += 1
    
    features["sentiment_polarity_variance"] = float(np.var(sent_polarities)) if len(sent_polarities) > 1 else 0.0
    features["neutral_sentence_ratio"] = neutral_count / len(sents) if sents else 0.0
    
    word_toks = [t for t in doc if _word_like(t)]
    if word_toks:
        positive_count = 0
        negative_count = 0
        for token in word_toks:
            word_blob = TextBlob(token.text.lower())
            polarity = word_blob.sentiment.polarity
            if polarity > 0.1:
                positive_count += 1
            elif polarity < -0.1:
                negative_count += 1
        features["positive_word_ratio"] = positive_count / len(word_toks)
        features["negative_word_ratio"] = negative_count / len(word_toks)
    else:
        features["positive_word_ratio"] = 0.0
        features["negative_word_ratio"] = 0.0
    
    return features


# pos features


def row_entropy(trans_counts: Counter) -> Dict[str, float]:
    # row-wise entropy H(P(.|A)) for each preceding tag
    res = {}
    row_totals = {A: 0 for A in POS_TAGS}
    for (A, B), v in trans_counts.items():
        if A in POS_SET and B in POS_SET:
            row_totals[A] += v
    for A in POS_TAGS:
        den = row_totals[A]
        if den == 0:
            res[A] = 0.0
            continue
        ent = 0.0
        for B in POS_TAGS:
            num = trans_counts.get((A, B), 0)
            if num == 0:
                continue
            p = num / den
            ent -= p * math.log2(max(p, 1e-12))
        res[A] = float(ent)
    return res

def run_lengths(seq: List[str], target: str) -> List[int]:
    # consecutive run lengths for target tag
    runs = []
    cur = 0
    for t in seq:
        if t == target:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    return runs

def pos_features(doc: Doc) -> Dict[str, float]:
    toks = [t for t in doc if not t.is_space]
    total_tokens = len(toks)
    
    if total_tokens == 0:
        feats = {f"pos_ratio_{tag}": 0.0 for tag in POS_TAGS}
        feats.update({
            "upos_entropy": 0.0, "pos_transition_entropy": 0.0, "pos_row_entropy_weighted": 0.0,
            "self_transition_rate": 0.0, "content_to_function_rate": 0.0, "function_to_content_rate": 0.0,
            "noun_verb_alternation_rate": 0.0, "content_function_ratio": 0.0,
            "noun_verb_ratio": 0.0, "adj_adv_ratio": 0.0,
            "verbs_per_100": 0.0, "nouns_per_100": 0.0, "adj_per_100": 0.0,
            "adv_per_100": 0.0, "pron_per_100": 0.0, "punct_per_100": 0.0,
            "tokens_per_sentence_mean": 0.0, "sentence_length_std": 0.0,
            "mean_nouns_per_sent": 0.0, "mean_verbs_per_sent": 0.0,
            "mean_adjs_per_sent": 0.0, "mean_advs_per_sent": 0.0,
            "prop_sents_with_verb": 0.0, "unique_upos_per_sent_mean": 0.0,
            "max_runlen_NOUN": 0.0, "max_runlen_PUNCT": 0.0
        })
        return feats
    
    pos_seq = [t.pos_ if t.pos_ in POS_SET else "X" for t in toks]
    pos_counts = Counter(pos_seq)
    
    # pos ratios
    feats = {f"pos_ratio_{tag}": safe_div(pos_counts.get(tag, 0), total_tokens) for tag in POS_TAGS}
    
    # unigram entropy
    feats["upos_entropy"] = shannon_entropy(pos_counts)
    
    # transitions
    transitions = list(zip(pos_seq, pos_seq[1:]))
    trans_counts = Counter(transitions)
    total_trans = sum(trans_counts.values())
    
    feats["pos_transition_entropy"] = shannon_entropy(trans_counts)
    
    # row entropy weighted by P(A)
    row_ent = row_entropy(trans_counts)
    pA = {A: safe_div(sum(v for (A2, _), v in trans_counts.items() if A2 == A), total_trans) for A in POS_TAGS}
    feats["pos_row_entropy_weighted"] = sum(pA[A] * row_ent.get(A, 0.0) for A in POS_TAGS)
    
    feats["self_transition_rate"] = safe_div(sum(trans_counts.get((A, A), 0) for A in POS_TAGS), total_trans)
    
    # content-function transitions
    c2f = sum(v for (A, B), v in trans_counts.items() if A in CONTENT_POS and B in FUNCTION_POS)
    f2c = sum(v for (A, B), v in trans_counts.items() if A in FUNCTION_POS and B in CONTENT_POS)
    feats["content_to_function_rate"] = safe_div(c2f, total_trans)
    feats["function_to_content_rate"] = safe_div(f2c, total_trans)
    
    # noun-verb alternation
    nv_alt = sum(trans_counts.get(pair, 0) for pair in [("NOUN","VERB"), ("VERB","NOUN"), ("NOUN","AUX"), ("AUX","NOUN")])
    feats["noun_verb_alternation_rate"] = safe_div(nv_alt, total_trans)
    
    # counts
    nouns = pos_counts.get("NOUN", 0)
    verbs = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
    adjs = pos_counts.get("ADJ", 0)
    advs = pos_counts.get("ADV", 0)
    prons = pos_counts.get("PRON", 0)
    punct = pos_counts.get("PUNCT", 0)
    
    content_sum = sum(pos_counts.get(t, 0) for t in CONTENT_POS)
    function_sum = sum(pos_counts.get(t, 0) for t in FUNCTION_POS)
    
    # ratios
    feats["content_function_ratio"] = safe_div(content_sum, function_sum)
    feats["noun_verb_ratio"] = safe_div(nouns, verbs)
    feats["adj_adv_ratio"] = safe_div(adjs, advs)
    
    # per-100 densities
    feats["verbs_per_100"] = 100.0 * safe_div(verbs, total_tokens)
    feats["nouns_per_100"] = 100.0 * safe_div(nouns, total_tokens)
    feats["adj_per_100"] = 100.0 * safe_div(adjs, total_tokens)
    feats["adv_per_100"] = 100.0 * safe_div(advs, total_tokens)
    feats["pron_per_100"] = 100.0 * safe_div(prons, total_tokens)
    feats["punct_per_100"] = 100.0 * safe_div(punct, total_tokens)
    
    # sentence-level aggregates
    sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
    if len(sents) == 0:
        sents = [doc]
    
    toks_per_sent = []
    nouns_ps, verbs_ps, adjs_ps, advs_ps, verb_presence = [], [], [], [], []
    unique_upos_counts = []
    
    for s in sents:
        stoks = [t for t in s if not t.is_space]
        toks_per_sent.append(len(stoks))
        upos_s = [t.pos_ if t.pos_ in POS_SET else "X" for t in stoks]
        c_s = Counter(upos_s)
        nouns_ps.append(c_s.get("NOUN", 0))
        verbs_ps.append(c_s.get("VERB", 0) + c_s.get("AUX", 0))
        adjs_ps.append(c_s.get("ADJ", 0))
        advs_ps.append(c_s.get("ADV", 0))
        verb_presence.append(1 if (c_s.get("VERB", 0) + c_s.get("AUX", 0)) > 0 else 0)
        unique_upos_counts.append(len({t for t in upos_s if t in POS_SET}))
    
    feats["tokens_per_sentence_mean"] = float(np.mean(toks_per_sent)) if toks_per_sent else 0.0
    feats["sentence_length_std"] = float(np.std(toks_per_sent, ddof=0)) if len(toks_per_sent) > 1 else 0.0
    feats["mean_nouns_per_sent"] = float(np.mean(nouns_ps)) if nouns_ps else 0.0
    feats["mean_verbs_per_sent"] = float(np.mean(verbs_ps)) if verbs_ps else 0.0
    feats["mean_adjs_per_sent"] = float(np.mean(adjs_ps)) if adjs_ps else 0.0
    feats["mean_advs_per_sent"] = float(np.mean(advs_ps)) if advs_ps else 0.0
    feats["prop_sents_with_verb"] = safe_div(sum(verb_presence), len(verb_presence))
    feats["unique_upos_per_sent_mean"] = float(np.mean(unique_upos_counts)) if unique_upos_counts else 0.0
    
    # run-length indicators
    rl_noun = run_lengths(pos_seq, "NOUN")
    rl_punct = run_lengths(pos_seq, "PUNCT")
    feats["max_runlen_NOUN"] = float(max(rl_noun)) if rl_noun else 0.0
    feats["max_runlen_PUNCT"] = float(max(rl_punct)) if rl_punct else 0.0
    
    return feats


# main feature extraction


def extract_stylometric_features(doc: Doc, text: str, cmu: dict, g2p: G2p) -> Dict[str, float]:
    features = {}
    
    sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
    word_toks = [t for t in doc if _word_like(t)]
    punct_toks = [t for t in doc if t.is_punct]
    nonspace_toks = [t for t in doc if not t.is_space]
    
    n_tokens = len(word_toks)
    n_sents = len(sents)
    
    features["n_tokens_doc"] = float(n_tokens)
    features["n_sentences_doc"] = float(n_sents)
    
    sent_word_counts = [sum(1 for t in sent if _word_like(t)) for sent in sents]
    features["avg_sentence_length"] = float(np.mean(sent_word_counts)) if sent_word_counts else np.nan
    features["sentence_length_std"] = float(np.std(sent_word_counts, ddof=0)) if len(sent_word_counts) > 1 else np.nan
    
    word_lengths = [len(t.text) for t in word_toks]
    features["avg_word_length"] = float(np.mean(word_lengths)) if word_lengths else np.nan
    
    vocab = {t.text.lower() for t in word_toks}
    features["type_token_ratio"] = len(vocab) / n_tokens if n_tokens > 0 else np.nan
    
    stop_count = sum(1 for t in word_toks if t.is_stop)
    features["stopword_ratio"] = stop_count / n_tokens if n_tokens > 0 else np.nan
    
    chars_punct = sum(len(t.text) for t in punct_toks)
    chars_nonspace = sum(len(t.text) for t in nonspace_toks)
    features["punctuation_ratio"] = chars_punct / chars_nonspace if chars_nonspace > 0 else np.nan
    
    syllable_counts = [syllables_hybrid(t.text, cmu, g2p) for t in word_toks]
    polysyllables = sum(1 for syl in syllable_counts if syl >= 3)
    chars_alnum = sum(_alnum_char_count(t.text) for t in nonspace_toks)
    chars_per_word = chars_alnum / n_tokens if n_tokens > 0 else 0.0
    
    readability = compute_readability(n_tokens, n_sents, syllable_counts, chars_per_word, polysyllables, polysyllables)
    features.update(readability)
    
    # ngram features with proper thresholds
    tok_win = [t.text.lower() for t in word_toks if _word_like(t)][:500]
    
    unigram = safe_ngram_stats(tok_win, n=1, min_ngrams=100)
    features["unigram_diversity"] = unigram["diversity"]
    features["unigram_entropy"] = unigram["entropy"]
    
    bigram = safe_ngram_stats(tok_win, n=2, min_ngrams=100)
    features["bigram_diversity"] = bigram["diversity"]
    features["bigram_entropy"] = bigram["entropy"]
    
    trigram = safe_ngram_stats(tok_win, n=3, min_ngrams=100)
    features["trigram_diversity"] = trigram["diversity"]
    features["trigram_entropy"] = trigram["entropy"]
    
    features["token_burstiness"] = calculate_burstiness(tok_win)
    
    char_div, char_ent = character_ngram_features(text, n=3)
    features["char_trigram_diversity"] = char_div
    features["char_trigram_entropy"] = char_ent
    features.update(character_statistics(text))
    features.update(compression_features(text))
    
    features.update(dependency_tree_features(doc))
    
    features["hapax_legomena_ratio"] = hapax_legomena_ratio(tok_win)
    features["hapax_type_ratio"] = hapax_type_ratio(tok_win)
    features["yules_k"] = yules_k(tok_win, min_tokens=100)
    
    features.update(punctuation_patterns(doc))
    features.update(pos_features(doc))
    
    return features


#general imputation function, for a full provided utils -> feature_utils.py can be utilisable and personable
#float must be setted to 0.2, 0.4 generally works well but is too generous, conservative heuristics are preferred (0.2)

def impute_missing_features(df: pd.DataFrame, max_missing_pct: float = 0.4) -> pd.DataFrame:
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    num_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['is_ai', 'id']]
    if not num_feats:
        return df
    
    # drop features with too much missing data
    missing_pct = df[num_feats].isna().mean()
    high_missing = missing_pct[missing_pct > max_missing_pct]
    
    if not high_missing.empty:
        print(f"\n[impute] dropping {len(high_missing)} features with >{max_missing_pct*100:.0f}% missing:")
        for feat, pct in high_missing.items():
            print(f"  - {feat}: {pct*100:.1f}% missing")
        num_feats = [f for f in num_feats if f not in high_missing.index]
        df.drop(columns=high_missing.index, inplace=True)
    
    if not num_feats:
        return df
    
    # stratified imputation by document length
    if "n_tokens_doc" in df.columns:
        bins = [0, 100, 250, 500, 10000]
        labels = ["S", "M", "L", "XL"]
        df["__len_bin__"] = pd.cut(df["n_tokens_doc"], bins=bins, right=False, labels=labels)
        
        for feat in num_feats:
            if df[feat].isna().any():
                group_medians = df.groupby("__len_bin__", observed=False)[feat].transform("median")
                df[feat] = df[feat].fillna(group_medians)
        
        df.drop(columns=["__len_bin__"], errors="ignore", inplace=True)
    
    # global median fallback
    for feat in num_feats:
        if df[feat].isna().any():
            median_val = df[feat].median()
            n_filled = df[feat].isna().sum()
            df[feat] = df[feat].fillna(median_val)
            if n_filled > 0:
                print(f"[impute] filled {n_filled} values in {feat} with median {median_val:.4f}")
    
    return df


# main pipeline


def extract_all_stylometric_features(df: pd.DataFrame, text_col: str, impute: bool = True) -> pd.DataFrame:
    print("\n" + "="*70)
    print("STYLOMETRIC FEATURE EXTRACTION PIPELINE")
    print("="*70)
    
    nlp, cmu, g2p = load_nlp_resources()
    
    texts = df[text_col].astype(str).tolist()
    doc_ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
    
    print(f"\nprocessing {len(texts)} documents...")
    
    all_features = []
    
    from tqdm import tqdm
    
    for i, doc in enumerate(tqdm(nlp.pipe(texts, batch_size=BATCH_SIZE), total=len(texts), desc="extracting")):
        text = texts[i]
        feats = extract_stylometric_features(doc, text, cmu, g2p)
        feats['id'] = doc_ids[i]
        all_features.append(feats)
        
        if DEVICE == "cuda" and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    features_df = pd.DataFrame(all_features)
    cols = ['id'] + [c for c in features_df.columns if c != 'id']
    features_df = features_df[cols]
    
    # imputation step
    if impute:
        print("\n[imputation] filling missing values...")
        features_df = impute_missing_features(features_df, max_missing_pct=0.4)
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"documents processed: {len(features_df)}")
    print(f"features extracted: {len(features_df.columns) - 1}")
    
    # nan summary
    nan_counts = features_df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\nremaining nan columns: {len(nan_cols)}")
        for col, count in nan_cols.items():
            print(f"  {col}: {count} ({100*count/len(features_df):.1f}%)")
    else:
        print("\nno nan values remaining")
    
    return features_df


# main entry point


def main():
    print("\n" + "="*70)
    print("STYLOMETRIC FEATURE EXTRACTOR")
    print("="*70)
    
    dataset_path = input("\nenter dataset path (csv with 'id' and text column): ").strip()
    
    if not Path(dataset_path).exists():
        print(f"error: file not found at {dataset_path}")
        return
    
    print(f"\nloading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"loaded {len(df)} documents")
    
    if 'id' not in df.columns:
        df['id'] = range(len(df))
        print("created 'id' column")
    
    if 'generation' in df.columns:
        text_col = 'generation'
    elif 'text' in df.columns:
        text_col = 'text'
    else:
        text_col = input("enter text column name: ").strip()
        if text_col not in df.columns:
            print(f"error: column '{text_col}' not found")
            return
    
    print(f"using text column: '{text_col}'")
    
    import time
    start_time = time.time()
    features_df = extract_all_stylometric_features(df, text_col, impute=True)
    elapsed = time.time() - start_time
    
    output_path = OUTPUT_PATH / "stylometric_features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"\nfeatures saved to: {output_path}")
    print(f"total time: {elapsed/60:.1f} minutes")
    
    print("\nsample output (first 3 rows):")
    print(features_df.head(3).to_string())
    
    return features_df


# test with mock data


def create_mock_data(n_samples: int = 20) -> pd.DataFrame:
    np.random.seed(42)
    
    #
    base_paragraphs = [
        """The advancement of artificial intelligence has transformed numerous industries across the global economy. 
        Machine learning algorithms now power recommendation systems, autonomous vehicles, and medical diagnostics. 
        Deep neural networks have achieved remarkable success in natural language processing and computer vision tasks. 
        Researchers continue to push the boundaries of what machines can accomplish through sophisticated architectures. 
        The ethical implications of these technologies remain a subject of intense debate among scholars and policymakers. 
        Companies invest billions of dollars annually in developing more capable and efficient AI systems. 
        The workforce must adapt to these changes through continuous learning and skill development programs.""",
        
        """Climate change represents one of the most pressing challenges facing humanity in the twenty-first century. 
        Rising global temperatures cause widespread disruption to ecosystems and weather patterns worldwide. 
        Coastal communities face increasing threats from sea level rise and more frequent extreme weather events. 
        Scientists have documented accelerating ice loss in both Arctic and Antarctic regions over recent decades. 
        International cooperation remains essential for implementing effective mitigation and adaptation strategies. 
        Renewable energy technologies offer promising pathways toward reducing greenhouse gas emissions significantly. 
        Individual actions combined with systemic changes can help address this existential environmental crisis.""",
        
        """The human brain contains approximately eighty-six billion neurons connected through trillions of synapses. 
        Neuroscientists study how these complex networks give rise to consciousness, memory, and behavior patterns. 
        Modern imaging techniques allow researchers to observe brain activity with unprecedented spatial resolution. 
        Understanding neural mechanisms has important implications for treating neurological and psychiatric disorders. 
        The field of computational neuroscience builds mathematical models to simulate brain function accurately. 
        Advances in brain-computer interfaces may eventually allow direct communication between minds and machines. 
        Memory formation involves intricate processes of encoding, consolidation, and retrieval across brain regions.""",
        
        """Economic inequality has increased substantially in many developed nations over the past several decades. 
        Wealth concentration among the top percentile raises concerns about social mobility and democratic governance. 
        Technological change and globalization have contributed to shifting labor market dynamics significantly. 
        Policy interventions such as progressive taxation and education investments aim to address these disparities. 
        The relationship between economic growth and inequality remains a contested topic among economists today. 
        Social safety nets provide crucial support for vulnerable populations during economic downturns and transitions. 
        Understanding the causes and consequences of inequality requires interdisciplinary research approaches.""",
        
        """Literature serves as a mirror reflecting the complexities of human experience across cultures and epochs. 
        Great works of fiction explore universal themes of love, loss, identity, and the search for meaning. 
        The novel emerged as a dominant literary form during the eighteenth century in European cultural contexts. 
        Postmodern authors challenge traditional narrative conventions through experimental techniques and structures. 
        Literary criticism examines how texts produce meaning through language, form, and cultural references. 
        Reading fiction has been shown to enhance empathy and emotional intelligence in psychological studies. 
        The digital age has transformed how literature is produced, distributed, and consumed by global audiences."""
    ]
    
    data = []
    for i in range(n_samples):
        # combine paragraphs for longer texts
        text = base_paragraphs[i % len(base_paragraphs)]
        if i % 2 == 0:
            text += " " + base_paragraphs[(i + 1) % len(base_paragraphs)]
        if i % 3 == 0:
            text += " Furthermore, these considerations have profound implications for future generations and their wellbeing."
        
        data.append({'id': f'doc_{i:04d}', 'text': text, 'is_ai': i % 2})
    
    return pd.DataFrame(data)

def run_test():
    print("\n" + "="*70)
    print("RUNNING TEST WITH MOCK DATA")
    print("="*70)
    
    mock_df = create_mock_data(20)
    mock_path = BASE_PATH / "mock_data.csv"
    mock_df.to_csv(mock_path, index=False)
    
    # show text lengths
    token_counts = mock_df['text'].apply(lambda x: len(x.split()))
    print(f"created mock data: {len(mock_df)} samples")
    print(f"token counts: min={token_counts.min()}, max={token_counts.max()}, mean={token_counts.mean():.0f}")
    print(f"saved to: {mock_path}")
    
    print("\nextracting features from mock data...")
    import time
    start_time = time.time()
    features_df = extract_all_stylometric_features(mock_df, 'text', impute=True)
    elapsed = time.time() - start_time
    
    test_output = OUTPUT_PATH / "test_stylometric_features.csv"
    features_df.to_csv(test_output, index=False)
    
    print("\n" + "="*70)
    print("TEST VALIDATION")
    print("="*70)
    
    n_cols = len(features_df.columns)
    print(f"columns extracted: {n_cols - 1} features + id")
    
    nan_cols = features_df.columns[features_df.isna().all()].tolist()
    partial_nan = features_df.columns[features_df.isna().any() & ~features_df.isna().all()].tolist()
    
    print(f"all-nan columns: {len(nan_cols)} {'PASS' if len(nan_cols) == 0 else 'FAIL: ' + str(nan_cols)}")
    print(f"partial-nan columns: {len(partial_nan)} (after imputation)")
    
    # feature categories
    pos_feats = [c for c in features_df.columns if c.startswith('pos_')]
    readability_feats = ['flesch_reading_ease', 'gunning_fog', 'smog_index', 'automated_readability_index']
    ngram_feats = [c for c in features_df.columns if 'gram' in c or 'diversity' in c or 'entropy' in c]
    
    print(f"\nfeature breakdown:")
    print(f"  pos features: {len(pos_feats)}")
    print(f"  readability features: {len([f for f in readability_feats if f in features_df.columns])}")
    print(f"  ngram/entropy features: {len(ngram_feats)}")
    
    print(f"\ntest completed in {elapsed:.1f} seconds")
    print(f"output saved to: {test_output}")
    
    print("\nfeature ranges (sample):")
    sample_cols = ['avg_word_length', 'type_token_ratio', 'flesch_reading_ease', 'yules_k', 'compression_ratio', 'upos_entropy', 'trigram_diversity']
    for col in sample_cols:
        if col in features_df.columns:
            min_v = features_df[col].min()
            max_v = features_df[col].max()
            nan_c = features_df[col].isna().sum()
            print(f"  {col}: [{min_v:.4f}, {max_v:.4f}] (nan: {nan_c})")
    
    return features_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test()
    else:
        main()