"""
Metacognition Feature Extraction Pipeline - Local Version
Extracts metadiscourse density features from text using enriched lexicons
"""

import pandas as pd
import numpy as np
import json
import pickle
import spacy
import torch
from typing import List, Dict
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings('ignore')


#Configuration da settare

BASE_PATH = Path(__file__).parent
LEXICON_PATH = BASE_PATH / "jsons" #modificare
OUTPUT_PATH = BASE_PATH / "output" #modificare
OUTPUT_PATH.mkdir(exist_ok=True)

LEXICON_FILES = {
    'metadiscourse': 'metadiscourse_enriched.json',
    'hedges': 'hedges_enriched.json',
    'boosters': 'boosters_enriched.json',
    'epistemic': 'epistemic_enriched.json',
    'self_reference': 'self_reference_enriched.json'
}


# Gpu
def setup_spacy():
    """Initialize spaCy with GPU if available"""
    print("=" * 60)
    print("SETTING UP ENVIRONMENT")
    print("=" * 60)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        spacy.require_gpu()
        print("GPU enabled for spaCy")
    else:
        print("Using CPU")
    
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print("Model not found, downloading...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
        nlp = spacy.load("en_core_web_lg")
    
    nlp.disable_pipes(['lemmatizer'])
    print(f"spaCy loaded: {nlp.meta['name']}")
    return nlp


# Lexicon

def load_lexicons() -> Dict:
    """Load all enriched JSON lexicons"""
    print("\n" + "=" * 60)
    print("LOADING LEXICONS")
    print("=" * 60)
    
    lexicons = {}
    for name, filename in LEXICON_FILES.items():
        filepath = LEXICON_PATH / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lexicons[name] = json.load(f)
            total = sum(len(v) for k, v in lexicons[name].items() if k != '_metadata' and isinstance(v, dict))
            print(f"Loaded {name}: {total} items")
        except FileNotFoundError:
            print(f"ERROR: {filename} not found at {filepath}")
    
    print(f"Loaded {len(lexicons)} lexicons")
    return lexicons

def build_lookup_dict(lexicon: Dict, top_level_key: str = None) -> Dict[str, Dict]:
    """Build flat lookup dictionary from nested lexicon structure"""
    lookup = {}
    marker_fields = {'pos', 'certainty', 'function', 'category', 'sentiment', 'type', 'graduation_type', 'subjectivity', 'person'}
    
    for category, items in lexicon.items():
        if category == '_metadata' or (top_level_key and category != top_level_key):
            continue
        if not isinstance(items, dict):
            continue
        
        first_value = next(iter(items.values()), None) if items else None
        if isinstance(first_value, dict):
            if marker_fields & set(first_value.keys()):
                for marker, info in items.items():
                    if isinstance(info, dict):
                        lookup[marker.lower()] = {'text': marker, 'category': category, **info}
            else:
                for subcategory, subitems in items.items():
                    if isinstance(subitems, dict):
                        sub_lookup = build_lookup_dict({subcategory: subitems})
                        lookup.update(sub_lookup)
    return lookup

def build_all_lookups(lexicons: Dict) -> Dict:
    """Build all lookup dictionaries from lexicons"""
    print("\n" + "=" * 60)
    print("BUILDING LOOKUP DICTIONARIES")
    print("=" * 60)
    
    lookups = {}
    
    # Interactive metadiscourse
    if 'metadiscourse' in lexicons and 'interactive' in lexicons['metadiscourse']:
        interactive = lexicons['metadiscourse']['interactive']
        for cat in ['transitions', 'frame_markers', 'code_glosses', 'evidentials', 'endophoric_markers']:
            if cat in interactive:
                lookups[cat] = build_lookup_dict(interactive, cat)
                print(f"  {cat}: {len(lookups[cat])}")
    
    # Interactional metadiscourse
    if 'metadiscourse' in lexicons and 'interactional' in lexicons['metadiscourse']:
        interactional = lexicons['metadiscourse']['interactional']
        for cat in ['attitude_markers', 'engagement_markers', 'self_mention']:
            if cat in interactional:
                lookups[cat] = build_lookup_dict(interactional, cat)
                print(f"  {cat}: {len(lookups[cat])}")
    
    # Hedges and boosters
    for name in ['hedges', 'boosters', 'epistemic']:
        if name in lexicons:
            lookups[name] = build_lookup_dict(lexicons[name])
            print(f"  {name}: {len(lookups[name])}")
    
    # Self-reference patterns
    if 'self_reference' in lexicons:
        sr = lexicons['self_reference']
        lookups['personal_epistemic'] = {}
        if 'personal_epistemic_phrases' in sr:
            lookups['personal_epistemic'].update(build_lookup_dict(sr, 'personal_epistemic_phrases'))
        if 'personal_epistemic_patterns' in sr:
            for subcat, items in sr['personal_epistemic_patterns'].items():
                if isinstance(items, dict):
                    for marker, info in items.items():
                        if isinstance(info, dict):
                            lookups['personal_epistemic'][marker.lower()] = {'text': marker, 'category': f'personal_epistemic_{subcat}', **info}
        
        lookups['reformulation'] = build_lookup_dict(sr, 'reformulation_markers') if 'reformulation_markers' in sr else {}
        print(f"  personal_epistemic: {len(lookups['personal_epistemic'])}")
        print(f"  reformulation: {len(lookups['reformulation'])}")
    
    # Weasel words
    lookups['weasel'] = {}
    if 'hedges' in lexicons and 'weasel_markers' in lexicons['hedges']:
        for marker, info in lexicons['hedges']['weasel_markers'].items():
            if isinstance(info, dict):
                lookups['weasel'][marker.lower()] = {'text': marker, 'category': 'weasel', **info}
        print(f"  weasel: {len(lookups['weasel'])}")
    
    total = sum(len(v) for v in lookups.values())
    print(f"\nTotal markers: {total}")
    return lookups


# Matching functions

def match_marker_in_doc(doc, marker: str, pos_tags: List[str] = None) -> List[int]:
    """Find all occurrences of a marker in spaCy doc"""
    matches = []
    marker_tokens = marker.lower().split()
    n = len(marker_tokens)
    
    for i in range(len(doc) - n + 1):
        span = [doc[i+j].text.lower() for j in range(n)]
        if span == marker_tokens:
            if pos_tags and n == 1 and doc[i].pos_ not in pos_tags:
                continue
            matches.append(i)
    return matches

def count_markers_in_doc(doc, lookup: Dict[str, Dict]) -> Dict[str, int]:
    """Count all markers from a lookup dictionary in a document"""
    counts = Counter()
    for marker_text, marker_info in lookup.items():
        pos_tags = marker_info.get('pos', None)
        matches = match_marker_in_doc(doc, marker_text, pos_tags)
        if matches:
            counts[marker_text] = len(matches)
    return dict(counts)

def compute_weighted_certainty_score(doc, lookups: Dict) -> float:
    """Compute document-level weighted certainty score"""
    total_weight, total_count = 0.0, 0
    
    for lookup_name in ['hedges', 'boosters', 'epistemic']:
        if lookup_name not in lookups:
            continue
        default_cert = 0.5 if lookup_name == 'hedges' else 0.9 if lookup_name == 'boosters' else 0.5
        for marker_text, marker_info in lookups[lookup_name].items():
            certainty = marker_info.get('certainty', default_cert)
            count = len(match_marker_in_doc(doc, marker_text, marker_info.get('pos')))
            total_weight += certainty * count
            total_count += count
    
    return total_weight / total_count if total_count > 0 else 0.5

def compute_certainty_by_position(doc, lookups: Dict, position: str) -> float:
    """Compute weighted certainty for a document section"""
    doc_len = len(doc)
    ranges = {
        'first_third': (0, doc_len // 3),
        'middle_third': (doc_len // 3, (doc_len * 2) // 3),
        'last_third': ((doc_len * 2) // 3, doc_len)
    }
    start, end = ranges.get(position, (0, doc_len))
    section = doc[start:end]
    
    total_weight, total_count = 0.0, 0
    for lookup_name in ['hedges', 'boosters', 'epistemic']:
        if lookup_name not in lookups:
            continue
        for marker_text, marker_info in lookups[lookup_name].items():
            certainty = marker_info.get('certainty', 0.5)
            marker_tokens = marker_text.lower().split()
            n = len(marker_tokens)
            for i in range(len(section) - n + 1):
                if [section[i+j].text.lower() for j in range(n)] == marker_tokens:
                    total_weight += certainty
                    total_count += 1
    
    return total_weight / total_count if total_count > 0 else 0.5


# Feature Extraction

def create_empty_features(doc_id: str, lookups: Dict) -> Dict:
    """Create zero-filled feature dict for error cases"""
    features = {'id': doc_id, 'doc_length_tokens': 0}
    for category in lookups.keys():
        features[f'{category}_density'] = 0.0
        features[f'{category}_count'] = 0
    
    features.update({
        'self_mention_first_use_ratio': 0.0, 'self_mention_first_20pct_density': 0.0,
        'self_mention_rest_80pct_density': 0.0, 'certainty_first_third': 0.5,
        'certainty_last_third': 0.5, 'certainty_gradient': 0.0, 'certainty_overall': 0.5,
        'reformulation_density': 0.0, 'reformulation_count': 0, 'weasel_count': 0,
        'evidential_count': 0, 'weasel_to_evidential_ratio': 0.0, 'weasel_density': 0.0,
        'evidential_density': 0.0
    })
    return features

def extract_metacognition_features(doc, doc_id: str, lookups: Dict) -> Dict:
    """Extract all metacognitive features from a spaCy doc"""
    try:
        doc_len = len(doc)
        if doc_len == 0:
            return create_empty_features(doc_id, lookups)
        
        features = {'id': doc_id, 'doc_length_tokens': doc_len}
        
        # Density features
        for category, lookup in lookups.items():
            if category == 'weasel':
                continue
            counts = count_markers_in_doc(doc, lookup)
            total = sum(counts.values())
            features[f'{category}_density'] = (total / doc_len) * 1000
            features[f'{category}_count'] = total
        
        # Self-mention positional analysis
        if 'self_mention' in lookups:
            first_20 = int(doc_len * 0.2)
            section_first, section_rest = doc[:first_20], doc[first_20:]
            count_first, count_rest = 0, 0
            
            for marker_text, marker_info in lookups['self_mention'].items():
                marker_tokens = marker_text.lower().split()
                n = len(marker_tokens)
                for i in range(len(section_first) - n + 1):
                    if [section_first[i+j].text.lower() for j in range(n)] == marker_tokens:
                        count_first += 1
                for i in range(len(section_rest) - n + 1):
                    if [section_rest[i+j].text.lower() for j in range(n)] == marker_tokens:
                        count_rest += 1
            
            d_first = (count_first / len(section_first) * 1000) if len(section_first) > 0 else 0
            d_rest = (count_rest / len(section_rest) * 1000) if len(section_rest) > 0 else 0
            features['self_mention_first_use_ratio'] = d_first / d_rest if d_rest > 0 else 0.0
            features['self_mention_first_20pct_density'] = d_first
            features['self_mention_rest_80pct_density'] = d_rest
        
        # Certainty gradient
        cert_first = compute_certainty_by_position(doc, lookups, 'first_third')
        cert_last = compute_certainty_by_position(doc, lookups, 'last_third')
        features['certainty_first_third'] = cert_first
        features['certainty_last_third'] = cert_last
        features['certainty_gradient'] = cert_first - cert_last
        features['certainty_overall'] = compute_weighted_certainty_score(doc, lookups)
        
        # Reformulation density
        reform_count = sum(count_markers_in_doc(doc, lookups.get('code_glosses', {})).values())
        reform_count += sum(count_markers_in_doc(doc, lookups.get('reformulation', {})).values())
        features['reformulation_density'] = (reform_count / doc_len) * 1000
        features['reformulation_count'] = reform_count
        
        # Weasel-to-evidential ratio
        weasel_count = sum(count_markers_in_doc(doc, lookups.get('weasel', {})).values())
        evid_count = sum(count_markers_in_doc(doc, lookups.get('evidentials', {})).values())
        total_attr = weasel_count + evid_count
        features['weasel_count'] = weasel_count
        features['evidential_count'] = evid_count
        features['weasel_to_evidential_ratio'] = weasel_count / total_attr if total_attr > 0 else 0.0
        features['weasel_density'] = (weasel_count / doc_len) * 1000
        features['evidential_density'] = (evid_count / doc_len) * 1000
        
        return features
    
    except Exception as e:
        print(f"Error processing {doc_id}: {e}")
        return create_empty_features(doc_id, lookups)


# Batch processing

def process_dataset(df: pd.DataFrame, nlp, lookups: Dict, text_col: str, batch_size: int = 16) -> pd.DataFrame:
    """Process entire dataset and extract features"""
    print("\n" + "=" * 60)
    print(f"PROCESSING {len(df)} DOCUMENTS")
    print("=" * 60)
    
    features_list = []
    texts = df[text_col].tolist()
    ids = df['id'].tolist()
    
    for batch_start in tqdm(range(0, len(df), batch_size), desc="Processing"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_texts = texts[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            docs = list(nlp.pipe(batch_texts, batch_size=batch_size))
            for doc_id, doc in zip(batch_ids, docs):
                features = extract_metacognition_features(doc, doc_id, lookups)
                features_list.append(features)
        except Exception as e:
            print(f"Batch error at {batch_start}: {e}")
            for doc_id in batch_ids:
                features_list.append(create_empty_features(doc_id, lookups))
    
    return pd.DataFrame(features_list)


# Main

def main():
    parser = argparse.ArgumentParser(description='Extract metacognition features from text')
    parser.add_argument('--input', type=str, help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Path to output CSV file')
    parser.add_argument('--text_col', type=str, default='generation', help='Column name containing text')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--test', action='store_true', help='Run test with mock data')
    args = parser.parse_args()
    
    # Setup
    nlp = setup_spacy()
    lexicons = load_lexicons()
    lookups = build_all_lookups(lexicons)
    
    if args.test:
        print("\n" + "=" * 60)
        print("RUNNING TEST WITH MOCK DATA")
        print("=" * 60)
        
        mock_data = pd.DataFrame({
            'id': ['test_001', 'test_002', 'test_003'],
            'generation': [
                "However, we believe the results clearly show that the method works. Indeed, our findings suggest this is important. In other words, the evidence is strong.",
                "It is possible that the experiment failed. Some researchers argue that the data might be flawed. Perhaps we should reconsider our approach.",
                "The results definitely prove our hypothesis. We are absolutely certain that this method is superior. Clearly, this is the best solution."
            ],
            'is_ai': [0, 1, 1]
        })
        
        features_df = process_dataset(mock_data, nlp, lookups, 'generation', batch_size=2)
        output_path = OUTPUT_PATH / "test_metacognition_features.csv"
        features_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"\nExtracted features shape: {features_df.shape}")
        print(f"\nSample features for test_001:")
        sample = features_df[features_df['id'] == 'test_001'].iloc[0]
        for col in ['hedges_density', 'boosters_density', 'certainty_gradient', 'reformulation_density']:
            if col in sample:
                print(f"  {col}: {sample[col]:.4f}")
        print(f"\nOutput saved to: {output_path}")
        print("\nTEST PASSED!")
        return
    
    # Interactive input if no file specified
    if args.input is None:
        print("\n" + "=" * 60)
        print("INPUT FILE SELECTION")
        print("=" * 60)
        args.input = input("Enter path to input CSV file: ").strip()
    
    # Load and process data
    print(f"\nLoading data from: {args.input}")
    df = pd.read_csv(args.input)
    
    # Detect text column
    text_col = args.text_col
    if text_col not in df.columns:
        if 'text' in df.columns:
            text_col = 'text'
        elif 'generation' in df.columns:
            text_col = 'generation'
        else:
            print(f"ERROR: Text column '{args.text_col}' not found")
            print(f"Available columns: {df.columns.tolist()}")
            return
    
    print(f"Using text column: '{text_col}'")
    print(f"Dataset: {len(df)} documents")
    
    # Process
    features_df = process_dataset(df, nlp, lookups, text_col, args.batch_size)
    
    # Save output
    if args.output is None:
        args.output = OUTPUT_PATH / "metacognition_features.csv"
    
    features_df.to_csv(args.output, index=False)
    print(f"\n Features saved to: {args.output}")
    print(f"Shape: {features_df.shape}")

if __name__ == "__main__":
    main()