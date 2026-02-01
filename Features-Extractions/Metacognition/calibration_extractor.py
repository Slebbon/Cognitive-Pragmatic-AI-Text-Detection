"""
Calibration Features Extraction Pipeline
Extracts 5 second-order features measuring alignment between linguistic uncertainty markers and statistical predictability.
"""

#CALIBRATION NEEDS Metacognition + Perplexity features as input; it is possible to directly run this script after 
#metacognition_extractor.py
#perplexity_extractor.py

import pandas as pd
import numpy as np
import pickle
import json
import time
import torch
from pathlib import Path
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Set, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


#Configuration


BASE_PATH = Path(__file__).parent
LEXICON_PATH = BASE_PATH / "jsons"
OUTPUT_PATH = BASE_PATH / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

MODEL_NAME = 'EleutherAI/pythia-160m'
MAX_LENGTH = 512
CHECKPOINT_INTERVAL = 50


#Lexicons loader


def extract_terms_from_json(json_path: Path, target_pos: Optional[Set[str]] = None) -> Set[str]:
    """Extract all terms from a lexicon JSON file"""
    terms = set()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for category_name, category_data in data.items():
            if category_name.startswith('_'):
                continue
            if isinstance(category_data, dict):
                for term, term_data in category_data.items():
                    if target_pos and isinstance(term_data, dict):
                        if 'pos' in term_data:
                            if not any(pos in target_pos for pos in term_data['pos']):
                                continue
                    terms.add(term.lower())
        return terms
    except Exception as e:
        print(f"[ERROR] Loading {json_path}: {e}")
        return set()


def load_lexicons(lexicon_path: Path) -> dict[str, Set[str]]:
    """Load all 5 lexicon files"""
    lexicon_files = {
        'hedges': 'hedges_enriched.json',
        'boosters': 'boosters_enriched.json',
        'epistemic': 'epistemic_enriched.json',
        'metadiscourse': 'metadiscourse_enriched.json',
        'self_reference': 'self_reference_enriched.json'
    }
    
    lexicons = {}
    print("\n[LOADING LEXICONS]")
    
    for name, filename in lexicon_files.items():
        filepath = lexicon_path / filename
        if filepath.exists():
            terms = extract_terms_from_json(filepath)
            lexicons[name] = terms
            print(f"  {name}: {len(terms)} terms")
        else:
            lexicons[name] = set()
            print(f"  {name}: NOT FOUND ({filename})")
    
    # Combine for aggregate sets used in features
    lexicons['all_hedges'] = lexicons['hedges'] | lexicons.get('epistemic', set())
    lexicons['all_boosters'] = lexicons['boosters']
    lexicons['all_metacog'] = lexicons['hedges'] | lexicons['boosters'] | lexicons['epistemic'] | lexicons['metadiscourse'] | lexicons['self_reference']
    
    print(f"\n  Combined sets:")
    print(f"    all_hedges (hedges+epistemic): {len(lexicons['all_hedges'])} terms")
    print(f"    all_boosters: {len(lexicons['all_boosters'])} terms")
    print(f"    all_metacog (all 5 combined): {len(lexicons['all_metacog'])} terms")
    
    return lexicons


#Caricamento Pythia


def load_model(model_name: str = MODEL_NAME) -> tuple:
    """Load Pythia model and tokenizer with GPU support"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    
    print(f"[LOADING] Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[LOADING] Model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


#Feature extraction


def compute_log_perplexity(text: str, model, tokenizer, device, max_length: int = MAX_LENGTH) -> float:
    """Compute log-perplexity for a text segment"""
    try:
        if len(text.split()) < 3:
            return np.nan
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding=False)
        input_ids = encodings.input_ids.to(device)
        if input_ids.shape[1] < 2:
            return np.nan
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        return outputs.loss.item()
    except Exception:
        return np.nan


def extract_sentence_densities(sentence: str, lexicon: Set[str]) -> float:
    """Extract density of markers from lexicon in sentence"""
    words = sentence.lower().split()
    if len(words) == 0:
        return 0.0
    count = sum(1 for word in words if word in lexicon)
    return count / len(words)


def compute_correlation_feature(densities: List[float], perplexities: List[float], min_sentences: int = 5) -> float:
    """Compute Pearson correlation between densities and perplexities"""
    if len(densities) < min_sentences or len(densities) != len(perplexities):
        return 0.0
    if np.all(np.array(densities) == 0) or np.var(perplexities) < 1e-6:
        return 0.0
    try:
        r, _ = pearsonr(densities, perplexities)
        return 0.0 if np.isnan(r) else r
    except Exception:
        return 0.0


def extract_calibration_features_for_doc(
    doc_id: str,
    sentences: List[str],
    merged_row: pd.Series,
    model, tokenizer, device,
    hedges: Set[str], boosters: Set[str],
    all_metacog: Set[str],
    max_length: int = MAX_LENGTH
) -> Dict:
    """Extract all 5 calibration features for a single document"""
    default_features = {
        'id': doc_id,
        'hedge_perplexity_correlation': 0.0,
        'booster_perplexity_anticorrelation': 0.0,
        'metacog_spike_perplexity_ratio': 1.0,
        'certainty_perplexity_alignment': 0.0,
        'reformulation_complexity_match': 0.0
    }
    
    if len(sentences) < 2:
        return default_features
    
    # Compute per-sentence perplexities and densities
    sentence_perplexities, hedge_densities, booster_densities, total_metacog_densities = [], [], [], []
    for sent in sentences:
        sentence_perplexities.append(compute_log_perplexity(sent, model, tokenizer, device, max_length))
        hedge_densities.append(extract_sentence_densities(sent, hedges))
        booster_densities.append(extract_sentence_densities(sent, boosters))
        total_metacog_densities.append(extract_sentence_densities(sent, all_metacog))
    
    # Filter valid sentences
    valid_indices = [i for i in range(len(sentences)) if not np.isnan(sentence_perplexities[i])]
    if len(valid_indices) < 2:
        return default_features
    
    valid_perp = [sentence_perplexities[i] for i in valid_indices]
    valid_hedge = [hedge_densities[i] for i in valid_indices]
    valid_booster = [booster_densities[i] for i in valid_indices]
    valid_total = [total_metacog_densities[i] for i in valid_indices]
    
    features = {'id': doc_id}
    
    # Feature 1: hedge_perplexity_correlation
    features['hedge_perplexity_correlation'] = compute_correlation_feature(valid_hedge, valid_perp)
    
    # Feature 2: booster_perplexity_anticorrelation
    features['booster_perplexity_anticorrelation'] = compute_correlation_feature(valid_booster, valid_perp)
    
    # Feature 3: metacog_spike_perplexity_ratio
    if len(valid_indices) >= 3:
        threshold_idx = max(1, int(0.8 * len(valid_indices)))
        sorted_indices = np.argsort(valid_total)[::-1]
        spike_indices = sorted_indices[:max(1, len(valid_indices) - threshold_idx)]
        if len(spike_indices) >= 3:
            spike_mean = np.mean([valid_perp[i] for i in spike_indices])
            baseline_mean = np.mean(valid_perp)
            features['metacog_spike_perplexity_ratio'] = min(spike_mean / baseline_mean, 2.5) if baseline_mean > 0 else 1.0
        else:
            features['metacog_spike_perplexity_ratio'] = 1.0
    else:
        features['metacog_spike_perplexity_ratio'] = 1.0
    
    # Feature 4: certainty_perplexity_alignment
    certainty_score = merged_row['certainty_overall']
    doc_log_perp = merged_row['doc_log_perplexity']
    perplexity_normalized = 1 / (1 + doc_log_perp)
    features['certainty_perplexity_alignment'] = 1 - abs(certainty_score - perplexity_normalized)
    
    # Feature 5: reformulation_complexity_match
    reformulation_density = merged_row['reformulation_density']
    sent_perp_variance = merged_row['sentence_log_perplexity_variance']
    features['reformulation_complexity_match'] = reformulation_density * sent_perp_variance if sent_perp_variance >= 0.01 else 0.0
    
    return features


#Main extraction pipeline

def run_extraction(
    metacog_path: Path,
    perplexity_path: Path,
    corpus_path: Path,
    sentence_seg_path: Path,
    output_path: Path,
    lexicon_path: Path,
    checkpoint_interval: int = CHECKPOINT_INTERVAL
) -> pd.DataFrame:
    """Main extraction pipeline"""
    print("=" * 80)
    print("CALIBRATION FEATURES EXTRACTION PIPELINE")
    print("=" * 80)
    
    # Load lexicons
    lexicons = load_lexicons(lexicon_path)
    hedges = lexicons['all_hedges']
    boosters = lexicons['all_boosters']
    all_metacog = lexicons['all_metacog']
    
    # Load data
    print("\n[LOADING] Metacognition features...")
    metacog_df = pd.read_csv(metacog_path)
    print(f"[OK] {metacog_df.shape[0]} rows")
    
    print("[LOADING] Perplexity features...")
    perplexity_df = pd.read_csv(perplexity_path)
    # Column name normalization
    rename_map = {
        'doc_perplexity': 'doc_log_perplexity',
        'mean_sentence_perplexity': 'mean_sentence_log_perplexity',
        'sentence_perplexity_variance': 'sentence_log_perplexity_variance'
    }
    perplexity_df.rename(columns={k: v for k, v in rename_map.items() if k in perplexity_df.columns}, inplace=True)
    print(f"[OK] {perplexity_df.shape[0]} rows")
    
    print("[LOADING] Corpus labels...")
    corpus_df = pd.read_csv(corpus_path)
    print(f"[OK] {corpus_df.shape[0]} rows")
    
    print("[LOADING] Sentence segmentation...")
    with open(sentence_seg_path, 'rb') as f:
        sentence_segmentation = pickle.load(f)
    print(f"[OK] {len(sentence_segmentation)} documents")
    
    # Merge data
    merged_df = metacog_df.merge(perplexity_df, on='id', how='inner')
    merged_df = merged_df.merge(corpus_df[['id', 'is_ai']], on='id', how='inner')
    print(f"\n[MERGED] {len(merged_df)} documents (AI: {(merged_df['is_ai']==1).sum()}, Human: {(merged_df['is_ai']==0).sum()})")
    
    # Load model
    model, tokenizer, device = load_model()
    
    # Extract features
    print(f"\n[EXTRACTING] Processing {len(merged_df)} documents...")
    checkpoint_path = output_path / "calibration_checkpoint.pkl"
    progress_path = output_path / "calibration_progress.txt"
    
    start_idx, features_list = 0, []
    if checkpoint_path.exists() and progress_path.exists():
        with open(checkpoint_path, 'rb') as f:
            features_list = pickle.load(f)
        with open(progress_path, 'r') as f:
            start_idx = int(f.read().strip())
        print(f"[RESUME] From document {start_idx}")
    
    start_time = time.time()
    error_count = 0
    
    for i in tqdm(range(start_idx, len(merged_df)), desc="Extracting"):
        row = merged_df.iloc[i]
        doc_id = row['id']
        try:
            sentences = sentence_segmentation.get(doc_id, [])
            features = extract_calibration_features_for_doc(
                doc_id, sentences, row, model, tokenizer, device, hedges, boosters, all_metacog
            )
            features_list.append(features)
            if torch.cuda.is_available() and (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"\n[ERROR] {doc_id}: {e}")
            features_list.append({'id': doc_id, 'hedge_perplexity_correlation': 0.0,
                'booster_perplexity_anticorrelation': 0.0, 'metacog_spike_perplexity_ratio': 1.0,
                'certainty_perplexity_alignment': 0.0, 'reformulation_complexity_match': 0.0})
        
        if (i + 1) % checkpoint_interval == 0:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(features_list, f)
            with open(progress_path, 'w') as f:
                f.write(str(i + 1))
    
    # Save final
    result_df = pd.DataFrame(features_list)
    final_path = output_path / "calibration_features_final.csv"
    result_df.to_csv(final_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] {len(result_df)} documents in {elapsed/60:.1f} min ({error_count} errors)")
    print(f"[SAVED] {final_path}")
    
    return result_df


if __name__ == "__main__":
    # Interactive path input
    print("=" * 80)
    print("CALIBRATION FEATURES EXTRACTION - LOCAL GPU")
    print("=" * 80)
    
    metacog_path = Path(input("Metacognition: ").strip())
    perplexity_path = Path(input("Perplexity:  ").strip())
    corpus_path = Path(input("Corpus CSV path (with is_ai column): ").strip())
    sentence_seg_path = Path(input("Sentence segmentation PKL path: ").strip())
    
    run_extraction(
        metacog_path=metacog_path,
        perplexity_path=perplexity_path,
        corpus_path=corpus_path,
        sentence_seg_path=sentence_seg_path,
        output_path=OUTPUT_PATH,
        lexicon_path=LEXICON_PATH
    )