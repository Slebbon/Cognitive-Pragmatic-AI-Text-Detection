# -*- coding: utf-8 -*-
"""
Coreference + REG Feature Extraction Pipeline for AI Text Detection
Local version with GPU support - Extracts 43 features across 6 tiers
"""

import os
import sys
import gc
import time
import pickle
import shutil
import argparse
from collections import Counter
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
import spacy
from fastcoref import FCoref
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION (!Settare i parametri qui!)

class Config:
    CHECKPOINT_INTERVAL = 500
    FASTCOREF_BATCH_SIZE = 16
    SPACY_BATCH_SIZE = 16
    LONG_RANGE_THRESHOLD = 3

#GPU 

def setup_device():
    """Setup GPU/CPU device"""
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return 'cuda:0'
    else:
        print("WARNING: No GPU detected, using CPU")
        return 'cpu'

def load_models(device: str):
    """Load FastCoref and spaCy models"""
    print("\nLoading models...")
    
    print("  Loading FastCoref...")
    coref_model = FCoref(device=device)
    print(f"  FastCoref loaded on {device}")
    
    print("  Loading spaCy...")
    nlp = spacy.load("en_core_web_trf", disable=["lemmatizer"])
    print(f"  spaCy pipeline: {nlp.pipe_names}")
    
    return coref_model, nlp

# CHAIN EXTRACTION

def get_sentence_index(doc, token_idx: int) -> int:
    """Get sentence index for token position"""
    for sent_idx, sent in enumerate(doc.sents):
        if sent.start <= token_idx < sent.end:
            return sent_idx
    return 0

def extract_chains_from_fastcoref(text: str, coref_model, nlp) -> Tuple[List, List, object]:
    """Extract coreference chains using FastCoref"""
    chains, all_mentions = [], []
    
    if not text or len(text.strip()) < 10:
        return chains, all_mentions, None
    
    try:
        preds = coref_model.predict(texts=[text])
        if not preds:
            return chains, all_mentions, None
        
        coref_result = preds[0]
        clusters = coref_result.get_clusters(as_strings=False)
        if not clusters:
            return chains, all_mentions, None
        
        doc = nlp(text)
        
        # Build char to token mapping
        char_to_token = {}
        for token in doc:
            for char_idx in range(token.idx, token.idx + len(token.text)):
                char_to_token[char_idx] = token.i
        
        for cluster in clusters:
            chain_mentions = []
            for char_start, char_end in cluster:
                start_token = char_to_token.get(char_start)
                end_token = char_to_token.get(char_end - 1)
                
                if start_token is None or end_token is None:
                    continue
                
                span = doc[start_token:end_token + 1]
                is_pronoun = span[0].pos_ == "PRON" if len(span) == 1 else span.root.pos_ == "PRON"
                
                mention = {
                    'text': span.text,
                    'start_token': start_token,
                    'end_token': end_token + 1,
                    'start_char': char_start,
                    'end_char': char_end,
                    'sent_idx': get_sentence_index(doc, start_token),
                    'is_pronoun': is_pronoun,
                    'token_count': len(span),
                    'span_start': start_token,
                    'span_end': end_token + 1
                }
                chain_mentions.append(mention)
                all_mentions.append(mention)
            
            if chain_mentions:
                chain_mentions.sort(key=lambda m: m['start_token'])
                chains.append(chain_mentions)
    
    except Exception as e:
        return chains, all_mentions, None
    
    return chains, all_mentions, doc

# BASELINE COREFERENCE FEATURES (6 features)

def calculate_pronoun_ratio(all_mentions: List[Dict]) -> float:
    if not all_mentions:
        return 0.0
    return sum(1 for m in all_mentions if m['is_pronoun']) / len(all_mentions)

def calculate_minimal_chain_ratio(chains: List[List[Dict]]) -> float:
    if not chains:
        return 0.0
    return sum(1 for c in chains if len(c) == 2) / len(chains)

def calculate_chain_length_stats(chains: List[List[Dict]]) -> Tuple[float, float]:
    if not chains:
        return 0.0, 0.0
    lengths = [len(c) for c in chains]
    return np.mean(lengths), np.var(lengths)

def calculate_long_range_ratio(chains: List[List[Dict]], threshold: int = 3) -> float:
    multi = [c for c in chains if len(c) >= 2]
    if not multi:
        return 0.0
    long_range = sum(1 for c in multi if max(m['sent_idx'] for m in c) - min(m['sent_idx'] for m in c) >= threshold)
    return long_range / len(multi)

def calculate_chain_connectivity(chains: List[List[Dict]], total_tokens: int) -> float:
    multi = [c for c in chains if len(c) >= 2]
    if not multi or total_tokens == 0:
        return 0.0
    
    scores = []
    for chain in multi:
        positions = [m['start_token'] for m in chain]
        coverage = min((max(positions) - min(positions)) / total_tokens, 1.0)
        
        if len(chain) <= 2:
            regularity = 1.0
        else:
            sorted_pos = sorted(positions)
            distances = [sorted_pos[i+1] - sorted_pos[i] for i in range(len(sorted_pos) - 1)]
            mean_d, std_d = np.mean(distances), np.std(distances)
            regularity = 1.0 / (1.0 + (std_d / mean_d if mean_d > 0 else 0))
        
        scores.append(coverage * regularity)
    
    return np.mean(scores)

# TIER 1: RMO FEATURES (3 features)

def calculate_tier1_rmo_features(chains: List[List[Dict]], doc) -> Dict[str, float]:
    """Repeat-Mention Overspecification"""
    features = {'repeat_mention_expansion_rate': 0.0, 'avg_tokens_added_on_repeat': 0.0, 'repeat_overspecification_ratio': 0.0}
    
    if not chains or doc is None:
        return features
    
    repeat_mentions, expansions, unnecessary = [], [], []
    
    for chain in chains:
        if len(chain) < 2:
            continue
        baseline_tokens = chain[0]['token_count']
        
        for i, mention in enumerate(chain[1:], 1):
            repeat_mentions.append(mention)
            if mention['token_count'] > baseline_tokens:
                expansion = mention['token_count'] - baseline_tokens
                expansions.append(expansion)
                prev = chain[i - 1]
                if not prev['is_pronoun'] and abs(mention['sent_idx'] - prev['sent_idx']) <= 1:
                    unnecessary.append(expansion)
    
    if repeat_mentions:
        features['repeat_mention_expansion_rate'] = len(expansions) / len(repeat_mentions)
        if expansions:
            features['avg_tokens_added_on_repeat'] = np.mean(expansions)
            features['repeat_overspecification_ratio'] = len(unnecessary) / len(expansions)
    
    return features

# TIER 2: MTA FEATURES (5 features)

def analyze_mention_modifications(mention: Dict, doc) -> Dict[str, int]:
    """Analyze syntactic modifications in mention"""
    mods = {'amod': 0, 'prep': 0, 'relcl': 0, 'compound': 0, 'poss': 0}
    span = doc[mention['span_start']:mention['span_end']]
    
    for token in span:
        if token.dep_ == 'amod':
            mods['amod'] += 1
        elif token.dep_ in ['prep', 'pobj']:
            mods['prep'] += 1
        elif token.dep_ == 'relcl':
            mods['relcl'] += 1
        elif token.dep_ == 'compound':
            mods['compound'] += 1
        elif token.dep_ in ['poss', 'nmod:poss']:
            mods['poss'] += 1
    
    return mods

def calculate_tier2_mta_features(chains: List[List[Dict]], all_mentions: List[Dict], doc) -> Dict[str, float]:
    """Modification Type Analysis"""
    features = {'adjective_modification_rate': 0.0, 'prepositional_modification_rate': 0.0,
                'relative_clause_rate': 0.0, 'modification_type_entropy': 0.0, 'avg_modifiers_per_mention': 0.0}
    
    if not all_mentions or doc is None:
        return features
    
    non_pronoun = [m for m in all_mentions if not m['is_pronoun']]
    if not non_pronoun:
        return features
    
    mod_counts, mod_types = [], []
    for m in non_pronoun:
        mods = analyze_mention_modifications(m, doc)
        mod_counts.append(sum(mods.values()))
        for k, v in mods.items():
            if v > 0:
                mod_types.append(k)
    
    n = len(non_pronoun)
    features['adjective_modification_rate'] = sum(1 for m in non_pronoun if analyze_mention_modifications(m, doc)['amod'] > 0) / n
    features['prepositional_modification_rate'] = sum(1 for m in non_pronoun if analyze_mention_modifications(m, doc)['prep'] > 0) / n
    features['relative_clause_rate'] = sum(1 for m in non_pronoun if analyze_mention_modifications(m, doc)['relcl'] > 0) / n
    
    if mod_types:
        counts = Counter(mod_types)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        features['modification_type_entropy'] = -sum(p * np.log2(p) for p in probs if p > 0)
    
    if mod_counts:
        features['avg_modifiers_per_mention'] = np.mean(mod_counts)
    
    return features

# SUPPLEMENTARY PRONOUN FEATURES (2 features)

def calculate_supplementary_pronoun_features(chains: List[List[Dict]]) -> Dict[str, float]:
    features = {'second_mention_pronoun_rate': 0.0, 'full_np_in_repeats_rate': 0.0}
    
    multi = [c for c in chains if len(c) >= 2]
    if not multi:
        return features
    
    second_pronoun = sum(1 for c in multi if c[1]['is_pronoun'])
    features['second_mention_pronoun_rate'] = second_pronoun / len(multi)
    
    repeat_count = sum(len(c) - 1 for c in multi)
    full_np = sum(1 for c in multi for m in c[1:] if not m['is_pronoun'])
    if repeat_count > 0:
        features['full_np_in_repeats_rate'] = full_np / repeat_count
    
    return features

# TIER 3: CONTEXT-SENSITIVE FEATURES (15 features)

def extract_entities_from_doc(doc) -> List[Dict]:
    return [{'text': e.text, 'start': e.start, 'end': e.end, 'label': e.label_, 'token_count': len(e)} for e in doc.ents]

def get_context_windows(doc, start: int, end: int) -> Dict[str, Tuple[int, int]]:
    """Extract multi-scale context windows"""
    doc_len = len(doc)
    all_sents = list(doc.sents)
    
    sent_idx = 0
    for i, sent in enumerate(all_sents):
        if sent.start <= start < sent.end:
            sent_idx = i
            break
    
    # Micro: +/- 1 sentence
    micro_start = all_sents[max(0, sent_idx - 1)].start
    micro_end = all_sents[min(len(all_sents) - 1, sent_idx + 1)].end
    
    # Meso: +/- 50 tokens
    meso_start, meso_end = max(0, start - 50), min(doc_len, end + 50)
    
    # Macro: paragraph boundaries (gap threshold = 3)
    macro_start_idx, macro_end_idx = sent_idx, sent_idx
    for i in range(sent_idx - 1, -1, -1):
        if all_sents[i + 1].start - all_sents[i].end >= 3:
            break
        macro_start_idx = i
    for i in range(sent_idx + 1, len(all_sents)):
        if all_sents[i].start - all_sents[i - 1].end >= 3:
            break
        macro_end_idx = i
    
    return {
        'micro': (micro_start, micro_end),
        'meso': (meso_start, meso_end),
        'macro': (all_sents[macro_start_idx].start, all_sents[macro_end_idx].end)
    }

def calculate_tier3_context_features(chains: List[List[Dict]], doc) -> Dict[str, float]:
    """Context-Sensitive Overspecification (15 features compressed)"""
    features = {
        'meso_avg_competitors_moderate': 0.0, 'macro_avg_context_entities': 0.0,
        'micro_logical_necessity_rate': 0.0, 'meso_pragmatic_necessity_rate': 0.0,
        'macro_overspecification_loose_rate': 0.0, 'meso_consensus_score_mean': 0.0,
        'micro_statistical_typicality_mean': 0.0, 'macro_modification_vs_competitors_ratio': 0.0,
        'meso_context_density': 0.0, 'micro_macro_competitor_ratio': 1.0,
        'scale_consistency_score': 1.0, 'micro_meso_overspec_gradient': 1.0,
        'meso_macro_context_ratio': 1.0, 'necessity_scale_variance': 0.0, 'competitor_scale_entropy': 0.0
    }
    
    if not chains or doc is None or len(doc) == 0:
        return features
    
    all_entities = extract_entities_from_doc(doc)
    scale_data = {
        'micro': {'comp': [], 'ent': [], 'log_nec': [], 'overspec': []},
        'meso': {'comp': [], 'ent': [], 'prag_nec': [], 'consensus': []},
        'macro': {'comp': [], 'ent': [], 'overspec': [], 'mod_vs_comp': []}
    }
    
    for chain in chains:
        for i, mention in enumerate(chain):
            start, end = mention.get('start_token', 0), mention.get('end_token', 0)
            if start >= len(doc) or end > len(doc):
                continue
            
            try:
                windows = get_context_windows(doc, start, end)
            except:
                continue
            
            for scale in ['micro', 'meso', 'macro']:
                ws, we = windows[scale]
                ctx_ents = [e for e in all_entities if ws <= e['start'] < we]
                competitors = [e for e in ctx_ents if e['start'] != start or e['end'] != end]
                
                scale_data[scale]['comp'].append(len(competitors))
                scale_data[scale]['ent'].append(len(ctx_ents))
            
            scale_data['micro']['log_nec'].append(1 if scale_data['micro']['comp'][-1] > 0 else 0)
            scale_data['micro']['overspec'].append(1 if scale_data['micro']['comp'][-1] == 0 and mention.get('token_count', 1) > 1 else 0)
            
            prag_nec = 1 if i == 0 or (start - chain[i-1].get('end_token', 0)) > 100 else 0
            scale_data['meso']['prag_nec'].append(prag_nec)
            scale_data['meso']['consensus'].append(1 if scale_data['meso']['comp'][-1] > 0 or prag_nec else 0)
            
            scale_data['macro']['overspec'].append(1 if scale_data['macro']['comp'][-1] == 0 and mention.get('token_count', 1) > 2 else 0)
            scale_data['macro']['mod_vs_comp'].append((mention.get('token_count', 1) - 1) / (scale_data['macro']['comp'][-1] + 1))
    
    # Aggregate features
    if scale_data['meso']['comp']:
        features['meso_avg_competitors_moderate'] = np.mean(scale_data['meso']['comp'])
    if scale_data['macro']['ent']:
        features['macro_avg_context_entities'] = np.mean(scale_data['macro']['ent'])
    if scale_data['micro']['log_nec']:
        features['micro_logical_necessity_rate'] = np.mean(scale_data['micro']['log_nec'])
    if scale_data['meso']['prag_nec']:
        features['meso_pragmatic_necessity_rate'] = np.mean(scale_data['meso']['prag_nec'])
    if scale_data['macro']['overspec']:
        features['macro_overspecification_loose_rate'] = np.mean(scale_data['macro']['overspec'])
    if scale_data['meso']['consensus']:
        features['meso_consensus_score_mean'] = np.mean(scale_data['meso']['consensus'])
    if scale_data['macro']['mod_vs_comp']:
        features['macro_modification_vs_competitors_ratio'] = np.mean(scale_data['macro']['mod_vs_comp'])
    
    features['meso_context_density'] = features['macro_avg_context_entities']
    
    # Cross-scale features
    micro_comp = np.mean(scale_data['micro']['comp']) if scale_data['micro']['comp'] else 0
    macro_comp = np.mean(scale_data['macro']['comp']) if scale_data['macro']['comp'] else 0
    if micro_comp > 0:
        features['micro_macro_competitor_ratio'] = macro_comp / micro_comp
    
    micro_os = np.mean(scale_data['micro']['overspec']) if scale_data['micro']['overspec'] else 0
    meso_os = np.mean([1 if c == 0 else 0 for c in scale_data['meso']['comp']]) if scale_data['meso']['comp'] else 0
    macro_os = features['macro_overspecification_loose_rate']
    features['scale_consistency_score'] = 1.0 - np.std([micro_os, meso_os, macro_os])
    
    if micro_os > 0:
        features['micro_meso_overspec_gradient'] = meso_os / micro_os
    
    meso_ctx = np.mean(scale_data['meso']['ent']) if scale_data['meso']['ent'] else 0
    if features['macro_avg_context_entities'] > 0:
        features['meso_macro_context_ratio'] = meso_ctx / features['macro_avg_context_entities']
    
    nec_vals = scale_data['micro']['log_nec'] + scale_data['meso']['prag_nec']
    if nec_vals:
        features['necessity_scale_variance'] = np.var(nec_vals)
    
    all_comp = scale_data['micro']['comp'] + scale_data['meso']['comp'] + scale_data['macro']['comp']
    if all_comp:
        counts = Counter(all_comp)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        features['competitor_scale_entropy'] = -sum(p * np.log2(p) for p in probs if p > 0)
    
    return features

# TIER 4: MINIMAL + SINGLETON FEATURES (12 features)

def count_modifiers_in_mention(mention: Dict, doc) -> Dict:
    try:
        span = doc[mention.get('start_token', 0):mention.get('end_token', 0)]
        mods = {'adjectives': 0, 'prepositional': 0, 'relative_clauses': 0}
        for token in span:
            if token.pos_ == 'ADJ':
                mods['adjectives'] += 1
            if token.dep_ in ['prep', 'pobj']:
                mods['prepositional'] += 1
            if token.dep_ == 'relcl':
                mods['relative_clauses'] += 1
        return {'num_modifiers': sum(mods.values()), 'modifier_breakdown': mods}
    except:
        return {'num_modifiers': 0, 'modifier_breakdown': {'adjectives': 0, 'prepositional': 0, 'relative_clauses': 0}}

def identify_singletons(all_entities: List[Dict], chains: List[List[Dict]]) -> List[Dict]:
    chain_spans = set()
    for chain in chains:
        for m in chain:
            chain_spans.add((m.get('start_token', 0), m.get('end_token', 0), m.get('text', '').lower()))
    
    singletons = []
    for e in all_entities:
        e_span = (e.get('start', 0), e.get('end', 0), e.get('text', '').lower())
        if not any(e_span[0] == c[0] and e_span[1] == c[1] or e_span[2] == c[2] for c in chain_spans):
            singletons.append(e)
    return singletons

def calculate_tier4_features(chains: List[List[Dict]], doc) -> Dict[str, float]:
    """Minimal Chains + Singletons (12 features)"""
    features = {
        'minimal_chain_count': 0, 'minimal_chain_ratio': 0.0, 'minimal_first_avg_modifiers': 0.0,
        'minimal_second_pronoun_rate': 0.0, 'minimal_avg_total_modifiers': 0.0,
        'singleton_count': 0, 'singleton_ratio': 0.0, 'singleton_avg_modifiers': 0.0,
        'singleton_descriptive_rate': 0.0, 'singleton_vs_chain_first_ratio': 0.0,
        'singleton_modification_entropy': 0.0, 'singleton_avg_tokens': 0.0
    }
    
    if not chains or doc is None:
        return features
    
    # Enrich with modifiers
    enriched = []
    for chain in chains:
        ec = []
        for m in chain:
            em = m.copy()
            em.update(count_modifiers_in_mention(m, doc))
            ec.append(em)
        if ec:
            enriched.append(ec)
    
    # Minimal chains
    minimal = [c for c in enriched if len(c) == 2]
    features['minimal_chain_count'] = len(minimal)
    if enriched:
        features['minimal_chain_ratio'] = len(minimal) / len(enriched)
    
    if minimal:
        features['minimal_first_avg_modifiers'] = np.mean([c[0].get('num_modifiers', 0) for c in minimal])
        features['minimal_second_pronoun_rate'] = sum(1 for c in minimal if c[1].get('is_pronoun')) / len(minimal)
        features['minimal_avg_total_modifiers'] = np.mean([c[0].get('num_modifiers', 0) + c[1].get('num_modifiers', 0) for c in minimal])
    
    # Singletons
    all_entities = extract_entities_from_doc(doc)
    singletons = identify_singletons(all_entities, chains)
    enriched_sing = [dict(s, **count_modifiers_in_mention(s, doc)) for s in singletons]
    
    all_chain_mentions = [m for c in enriched for m in c]
    total_entities = len(enriched_sing) + len(all_chain_mentions)
    
    features['singleton_count'] = len(enriched_sing)
    if total_entities > 0:
        features['singleton_ratio'] = len(enriched_sing) / total_entities
    
    if enriched_sing:
        features['singleton_avg_modifiers'] = np.mean([s.get('num_modifiers', 0) for s in enriched_sing])
        features['singleton_descriptive_rate'] = sum(1 for s in enriched_sing if s.get('num_modifiers', 0) >= 2) / len(enriched_sing)
        
        if enriched:
            first_mods = np.mean([c[0].get('num_modifiers', 0) for c in enriched])
            if first_mods > 0:
                features['singleton_vs_chain_first_ratio'] = features['singleton_avg_modifiers'] / first_mods
        
        mod_types = []
        for s in enriched_sing:
            bd = s.get('modifier_breakdown', {})
            if bd.get('adjectives', 0) > 0:
                mod_types.append('adj')
            if bd.get('prepositional', 0) > 0:
                mod_types.append('prep')
            if bd.get('relative_clauses', 0) > 0:
                mod_types.append('relcl')
        
        if mod_types:
            counts = Counter(mod_types)
            total = len(mod_types)
            probs = [c / total for c in counts.values()]
            features['singleton_modification_entropy'] = -sum(p * np.log2(p) for p in probs if p > 0)
        
        features['singleton_avg_tokens'] = np.mean([s.get('token_count', 1) for s in enriched_sing])
    
    return features

#Master Extraction

def extract_all_features(text: str, coref_model, nlp) -> Dict[str, float]:
    """Extract all 43 features from a single text"""
    features = {
        'pronoun_ratio': 0.0, 'minimal_chain_ratio': 0.0, 'avg_chain_length': 0.0,
        'chain_length_variance': 0.0, 'long_range_coref_ratio': 0.0, 'chain_connectivity': 0.0,
        'repeat_mention_expansion_rate': 0.0, 'avg_tokens_added_on_repeat': 0.0, 'repeat_overspecification_ratio': 0.0,
        'adjective_modification_rate': 0.0, 'prepositional_modification_rate': 0.0, 'relative_clause_rate': 0.0,
        'modification_type_entropy': 0.0, 'avg_modifiers_per_mention': 0.0,
        'second_mention_pronoun_rate': 0.0, 'full_np_in_repeats_rate': 0.0,
        'meso_avg_competitors_moderate': 0.0, 'macro_avg_context_entities': 0.0,
        'micro_logical_necessity_rate': 0.0, 'meso_pragmatic_necessity_rate': 0.0,
        'macro_overspecification_loose_rate': 0.0, 'meso_consensus_score_mean': 0.0,
        'micro_statistical_typicality_mean': 0.0, 'macro_modification_vs_competitors_ratio': 0.0,
        'meso_context_density': 0.0, 'micro_macro_competitor_ratio': 1.0,
        'scale_consistency_score': 1.0, 'micro_meso_overspec_gradient': 1.0,
        'meso_macro_context_ratio': 1.0, 'necessity_scale_variance': 0.0, 'competitor_scale_entropy': 0.0,
        'minimal_chain_count': 0, 'minimal_chain_ratio': 0.0, 'minimal_first_avg_modifiers': 0.0,
        'minimal_second_pronoun_rate': 0.0, 'minimal_avg_total_modifiers': 0.0,
        'singleton_count': 0, 'singleton_ratio': 0.0, 'singleton_avg_modifiers': 0.0,
        'singleton_descriptive_rate': 0.0, 'singleton_vs_chain_first_ratio': 0.0,
        'singleton_modification_entropy': 0.0, 'singleton_avg_tokens': 0.0
    }
    
    if not text or len(text.strip()) < 10:
        return features
    
    try:
        chains, all_mentions, doc = extract_chains_from_fastcoref(text, coref_model, nlp)
        if not all_mentions or doc is None:
            return features
        
        features['pronoun_ratio'] = calculate_pronoun_ratio(all_mentions)
        features['minimal_chain_ratio'] = calculate_minimal_chain_ratio(chains)
        avg_len, var = calculate_chain_length_stats(chains)
        features['avg_chain_length'], features['chain_length_variance'] = avg_len, var
        features['long_range_coref_ratio'] = calculate_long_range_ratio(chains)
        features['chain_connectivity'] = calculate_chain_connectivity(chains, len(doc))
        
        features.update(calculate_tier1_rmo_features(chains, doc))
        features.update(calculate_tier2_mta_features(chains, all_mentions, doc))
        features.update(calculate_supplementary_pronoun_features(chains))
        features.update(calculate_tier3_context_features(chains, doc))
        features.update(calculate_tier4_features(chains, doc))
    
    except Exception as e:
        pass
    
    return features

def create_empty_features() -> Dict[str, float]:
    """Create zero-filled features for error cases"""
    return extract_all_features("", None, None)

#Extraction

def save_checkpoint(data, path: str, progress_path: str, idx: int):
    tmp_ckpt, tmp_prog = path + '.tmp', progress_path + '.tmp'
    with open(tmp_ckpt, 'wb') as f:
        pickle.dump(data, f)
    with open(tmp_prog, 'w') as f:
        f.write(str(idx))
    shutil.move(tmp_ckpt, path)
    shutil.move(tmp_prog, progress_path)

def run_batch_extraction(df: pd.DataFrame, text_col: str, coref_model, nlp, output_dir: str) -> pd.DataFrame:
    """Run batch feature extraction with checkpointing"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'coref_features.pkl')
    progress_path = os.path.join(output_dir, 'coref_progress.txt')
    
    features_list, start_idx = [], 0
    if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
        with open(checkpoint_path, 'rb') as f:
            features_list = pickle.load(f)
        with open(progress_path, 'r') as f:
            start_idx = int(f.read().strip())
        print(f"Resuming from index {start_idx}")
    
    print(f"\nExtracting features from {len(df)} documents...")
    
    error_count = 0
    start_time = time.time()
    
    for i in tqdm(range(start_idx, len(df)), desc="Feature extraction"):
        text = str(df[text_col].iloc[i]) if pd.notna(df[text_col].iloc[i]) else ""
        
        try:
            feats = extract_all_features(text, coref_model, nlp)
            feats['index'] = i
            features_list.append(feats)
        except Exception as e:
            error_count += 1
            feats = create_empty_features()
            feats['index'] = i
            features_list.append(feats)
        
        if torch.cuda.is_available() and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
        
        if (i + 1) % Config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(features_list, checkpoint_path, progress_path, i + 1)
            gc.collect()
    
    # Create result
    features_df = pd.DataFrame(features_list)
    result_df = df.copy()
    
    for col in features_df.columns:
        if col != 'index':
            result_df[col] = 0.0
            result_df.loc[features_df['index'], col] = features_df[col].values
    
    output_path = os.path.join(output_dir, 'coreference_features_final.csv')
    result_df.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"\nExtraction complete: {len(result_df)} documents in {elapsed/60:.1f} minutes")
    print(f"Errors: {error_count}")
    print(f"Output: {output_path}")
    
    return result_df

# Main

def main():
    parser = argparse.ArgumentParser(description='Coreference Feature Extraction Pipeline')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--text-column', type=str, default='generation', help='Name of text column')
    parser.add_argument('--output', type=str, default='./coreference_output', help='Output directory')
    args = parser.parse_args()
    
    print("="*60)
    print("COREFERENCE FEATURE EXTRACTION")
    print("="*60)
    
    device = setup_device()
    coref_model, nlp = load_models(device)
    
    print(f"\nLoading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"Loaded {len(df)} documents")
    
    if args.text_column not in df.columns:
        raise ValueError(f"Text column '{args.text_column}' not found")
    
    result_df = run_batch_extraction(df, args.text_column, coref_model, nlp, args.output)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Features extracted: 43")
    print(f"Output: {args.output}")

if __name__ == '__main__':
    main()