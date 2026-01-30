# -*- coding: utf-8 -*-
"""
Temporal Reasoning Feature Extraction Pipeline for AI Text Detection
Local version - extracts 32 temporal features from pre-extracted components
"""

import os
import sys
import gc
import time
import pickle
import shutil
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import entropy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Da settare! --> sulla base della GPU (x colab più bassa)


class Config:
    CHECKPOINT_INTERVAL = 20
    BATCH_SIZE = 20
    MAX_EVENTS_FOR_CSP = 150
    MAX_NODES_FOR_HEAVY_GRAPH = 200
    MAX_NODES_FOR_EXACT_CYCLES = 120
    MAX_EDGES_FOR_EXACT_CYCLES = 800
    MAX_CYCLES_ENUM = 1000
    MAX_SAMPLE_START_NODES = 50
    MAX_CYCLE_LENGTH = 4
    MAX_SAMPLE_STEPS = 5000


# utility functions


def compute_entropy(labels):
    """Shannon entropy of label distribution"""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c/total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def safe_division(numerator, denominator, default=0.0):
    """Safe division with default value"""
    return numerator / denominator if denominator != 0 else default

def save_checkpoint(data, checkpoint_path: str, progress_path: str, progress_idx: int):
    """Atomic checkpoint save"""
    tmp_ckpt = checkpoint_path + '.tmp'
    tmp_prog = progress_path + '.tmp'
    with open(tmp_ckpt, 'wb') as f:
        pickle.dump(data, f)
    with open(tmp_prog, 'w') as f:
        f.write(str(progress_idx))
    shutil.move(tmp_ckpt, checkpoint_path)
    shutil.move(tmp_prog, progress_path)

def load_checkpoint(checkpoint_path: str, progress_path: str):
    """Load checkpoint if exists"""
    if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        with open(progress_path, 'r') as f:
            start_idx = int(f.read().strip())
        return data, start_idx
    return [], 0


# LOAD


def load_temporal_components(extraction_path: str) -> Tuple[Dict, Dict, Dict, Dict, Dict, str]:
    """Load all pre-extracted temporal components"""
    print(f"\nLoading temporal components from: {extraction_path}")
    
    def find_file(base, candidates, desc):
        for c in candidates:
            p = os.path.join(base, c)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"{desc} not found in {base}")
    
    # Load events
    events_path = find_file(extraction_path, 
        ['events/stanza_events_final.pkl', 'stanza_events_final.pkl'], 'Events')
    with open(events_path, 'rb') as f:
        events_dict = pickle.load(f)
    print(f"  Events: {len(events_dict)} documents")
    
    # Load TIMEX
    timex_path = find_file(extraction_path,
        ['timex/sutime_timex_final.pkl', 'sutime_timex_final.pkl'], 'TIMEX')
    with open(timex_path, 'rb') as f:
        timex_dict = pickle.load(f)
    print(f"  TIMEX: {len(timex_dict)} documents")
    
    # Load relations
    relations_path = find_file(extraction_path,
        ['relations/binary_relations_final.pkl', 'binary_relations_final.pkl'], 'Relations')
    with open(relations_path, 'rb') as f:
        relations_dict = pickle.load(f)
    print(f"  Relations: {len(relations_dict)} documents")
    
    # Load raw graphs
    raw_path = find_file(extraction_path,
        ['graphs/temporal_graphs_raw_final.pkl', 'temporal_graphs_final.pkl'], 'Raw graphs')
    with open(raw_path, 'rb') as f:
        raw_graphs = pickle.load(f)
    print(f"  Raw graphs: {len(raw_graphs)} documents")
    
    # Load DAG graphs (prefer greedy over ILP)
    dag_graphs, dag_type = None, 'Raw'
    for path, dtype in [
        ('graphs/greedy_temporal_graphs_final.pkl', 'Greedy'),
        ('graphs/ilp_temporal_graphs_final.pkl', 'ILP')
    ]:
        full_path = os.path.join(extraction_path, path)
        if os.path.exists(full_path):
            with open(full_path, 'rb') as f:
                dag_graphs = pickle.load(f)
            dag_type = dtype
            break
    
    if dag_graphs is None:
        dag_graphs = raw_graphs
        dag_type = 'Raw'
    print(f"  DAG graphs ({dag_type}): {len(dag_graphs)} documents")
    
    return events_dict, timex_dict, relations_dict, raw_graphs, dag_graphs, dag_type

def load_dataset(dataset_path: str) -> Dict[str, str]:
    """Load dataset and create texts dictionary"""
    df = pd.read_csv(dataset_path)
    text_col = 'generation' if 'generation' in df.columns else 'text'
    if text_col not in df.columns:
        raise ValueError("No text column found")
    texts_dict = {row['id']: row[text_col] for _, row in df.iterrows()}
    print(f"  Texts: {len(texts_dict)} documents")
    return texts_dict


# Event-level features


def extract_event_structure_features(doc_id, events, timex_list, text) -> Dict:
    """Extract 6 event structure features"""
    features = {}
    n_sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
    n_events = len(events)
    n_timex = len(timex_list)
    
    features['temp_num_events'] = n_events
    features['temp_events_per_sentence'] = safe_division(n_events, n_sentences)
    features['temp_num_timex'] = n_timex
    features['temp_timex_event_ratio'] = safe_division(n_timex, n_events)
    
    if events:
        lemmas = [e['lemma'] for e in events]
        features['temp_event_lexical_diversity'] = len(set(lemmas)) / len(lemmas)
        tenses = [e.get('tense', 'None') for e in events]
        features['temp_tense_distribution_entropy'] = compute_entropy(tenses)
    else:
        features['temp_event_lexical_diversity'] = 0.0
        features['temp_tense_distribution_entropy'] = 0.0
    
    return features


# Relation-level features


def compute_transitivity_violations_fast(relations) -> int:
    """Count transitivity violations with set operations"""
    if not relations or len(relations) < 3:
        return 0
    
    adj = defaultdict(set)
    for r in relations:
        if r['relation'] == 'BEFORE':
            adj[r['event1']].add(r['event2'])
        elif r['relation'] == 'AFTER':
            adj[r['event2']].add(r['event1'])
    
    if len(relations) < 300:
        violations = 0
        for a, successors_a in adj.items():
            for b in successors_a:
                if b in adj:
                    violations += len(adj[b] - successors_a)
        return violations
    else:
        import random
        sample_size = min(100, len(adj))
        sampled_keys = random.sample(list(adj.keys()), sample_size)
        violations = 0
        for a in sampled_keys:
            successors_a = adj[a]
            for b in list(successors_a)[:10]:
                if b in adj:
                    violations += len(adj[b] - successors_a)
        return int(violations * (len(adj) / sample_size) ** 2)

def approximate_cycle_features(raw_graph, relations, random_state=42) -> Tuple[float, float]:
    """Bounded DFS sampling for cycle detection"""
    nodes = list(raw_graph.nodes())
    if not nodes or not relations:
        return 0.0, 0.0
    
    import random
    rng = random.Random(random_state)
    start_nodes = nodes if len(nodes) <= Config.MAX_SAMPLE_START_NODES else rng.sample(nodes, Config.MAX_SAMPLE_START_NODES)
    
    cycle_count, cycle_edges, steps = 0, set(), 0
    
    for start in start_nodes:
        stack = [(start, [start])]
        while stack and steps < Config.MAX_SAMPLE_STEPS:
            steps += 1
            node, path = stack.pop()
            if len(path) > Config.MAX_CYCLE_LENGTH:
                continue
            for nbr in raw_graph.successors(node):
                if nbr == path[0] and len(path) >= 3:
                    cycle_count += 1
                    for u, v in zip(path, path[1:]):
                        cycle_edges.add((u, v))
                    cycle_edges.add((path[-1], path[0]))
                elif nbr not in path and len(path) < Config.MAX_CYCLE_LENGTH:
                    stack.append((nbr, path + [nbr]))
            if steps >= Config.MAX_SAMPLE_STEPS:
                break
        if steps >= Config.MAX_SAMPLE_STEPS:
            break
    
    return float(cycle_count), safe_division(len(cycle_edges), len(relations))

def extract_relation_features(relations, raw_graph) -> Dict:
    """Extract 9 relation-level features"""
    if not relations:
        return {
            'temp_rel_mean_confidence': 0.0, 'temp_rel_confidence_variance': 0.0,
            'temp_rel_before_after_ratio': 0.0, 'temp_rel_raw_cycle_count': 0.0,
            'temp_rel_cycle_edge_ratio': 0.0, 'temp_rel_parallel_edge_rate': 0.0,
            'temp_rel_type_entropy': 0.0, 'temp_rel_transitivity_violation_rate': 0.0,
            'temp_rel_cycle_approx_flag': 0.0
        }
    
    features = {}
    confidences, relation_types = [], []
    edge_pairs = defaultdict(int)
    
    for r in relations:
        confidences.append(r['confidence'])
        relation_types.append(r['relation'])
        pair = tuple(sorted([r['event1'], r['event2']]))
        edge_pairs[pair] += 1
    
    features['temp_rel_mean_confidence'] = float(np.mean(confidences))
    features['temp_rel_confidence_variance'] = float(np.var(confidences))
    
    before_count = relation_types.count('BEFORE')
    after_count = relation_types.count('AFTER')
    features['temp_rel_before_after_ratio'] = safe_division(before_count, after_count, 1.0)
    
    parallel_count = sum(1 for c in edge_pairs.values() if c > 1)
    features['temp_rel_parallel_edge_rate'] = safe_division(parallel_count, len(edge_pairs))
    features['temp_rel_type_entropy'] = compute_entropy(relation_types)
    
    # Cycle features
    cycle_count, approx_flag = 0.0, 0.0
    cycle_edges = set()
    
    if raw_graph is not None:
        n_nodes, n_edges = raw_graph.number_of_nodes(), raw_graph.number_of_edges()
        if (n_nodes <= Config.MAX_NODES_FOR_EXACT_CYCLES and 
            n_edges <= Config.MAX_EDGES_FOR_EXACT_CYCLES):
            try:
                for i, cyc in enumerate(nx.simple_cycles(raw_graph)):
                    if i >= Config.MAX_CYCLES_ENUM:
                        break
                    cycle_count += 1
                    for u, v in zip(cyc, cyc[1:] + [cyc[0]]):
                        cycle_edges.add((u, v))
            except:
                pass
        else:
            cycle_count, features['temp_rel_cycle_edge_ratio'] = approximate_cycle_features(raw_graph, relations)
            approx_flag = 1.0
    
    if 'temp_rel_cycle_edge_ratio' not in features:
        features['temp_rel_cycle_edge_ratio'] = safe_division(len(cycle_edges), len(relations))
    
    features['temp_rel_raw_cycle_count'] = float(cycle_count)
    features['temp_rel_cycle_approx_flag'] = approx_flag
    
    violations = compute_transitivity_violations_fast(relations)
    features['temp_rel_transitivity_violation_rate'] = safe_division(violations, len(relations))
    
    return features


# Graph-theoretic features


def extract_graph_features(dag_graph, raw_graph) -> Dict:
    """Extract 9 graph-theoretic features"""
    if dag_graph is None or dag_graph.number_of_nodes() == 0:
        return {
            'tg_edge_retention': 0.0, 'tg_degree_entropy': 0.0,
            'tg_avg_in_degree': 0.0, 'tg_avg_out_degree': 0.0,
            'tg_ordering_entropy': 0.0, 'tg_longest_path': 0,
            'tg_mean_depth': 0.0, 'tg_branching_factor': 0.0,
            'tg_global_coherence': 0.0
        }
    
    features = {}
    n_nodes = dag_graph.number_of_nodes()
    n_edges = dag_graph.number_of_edges()
    
    # Edge retention
    if 'edge_retention_rate' in dag_graph.graph:
        features['tg_edge_retention'] = dag_graph.graph['edge_retention_rate']
    else:
        raw_edges = raw_graph.number_of_edges() if raw_graph else 0
        features['tg_edge_retention'] = safe_division(n_edges, raw_edges)
    
    # Degree statistics
    nodes = list(dag_graph.nodes())
    in_degrees = [d for _, d in dag_graph.in_degree()]
    out_degrees = [d for _, d in dag_graph.out_degree()]
    
    features['tg_degree_entropy'] = compute_entropy(in_degrees + out_degrees)
    features['tg_avg_in_degree'] = float(np.mean(in_degrees)) if in_degrees else 0.0
    features['tg_avg_out_degree'] = float(np.mean(out_degrees)) if out_degrees else 0.0
    
    # Topological ordering entropy
    try:
        topo_list = list(nx.topological_sort(dag_graph))
        features['tg_ordering_entropy'] = compute_entropy(list(range(len(topo_list))))
    except:
        features['tg_ordering_entropy'] = 0.0
    
    # Longest path
    try:
        features['tg_longest_path'] = nx.dag_longest_path_length(dag_graph)
    except:
        features['tg_longest_path'] = 0
    
    # Mean depth via BFS
    sources = [n for n, d in dag_graph.in_degree() if d == 0]
    if sources and n_nodes < Config.MAX_NODES_FOR_HEAVY_GRAPH:
        depths = {s: 0 for s in sources}
        visited, queue = set(sources), list(sources)
        while queue:
            node = queue.pop(0)
            for succ in dag_graph.successors(node):
                if succ not in visited:
                    depths[succ] = depths[node] + 1
                    visited.add(succ)
                    queue.append(succ)
        features['tg_mean_depth'] = float(np.mean(list(depths.values())))
    else:
        features['tg_mean_depth'] = 0.0
    
    # Branching factor
    non_zero_out = [d for d in out_degrees if d > 0]
    features['tg_branching_factor'] = float(np.mean(non_zero_out)) if non_zero_out else 0.0
    
    # Global coherence
    if n_nodes < Config.MAX_NODES_FOR_HEAVY_GRAPH:
        try:
            reachable = sum(len(nx.descendants(dag_graph, n)) for n in nodes)
            features['tg_global_coherence'] = safe_division(n_edges, reachable)
        except:
            features['tg_global_coherence'] = 0.0
    else:
        features['tg_global_coherence'] = 0.0
    
    return features


# constraint satisfaction based features


def compute_simple_violations(relations) -> int:
    """Count antisymmetry violations"""
    if not relations:
        return 0
    edges, violations = set(), 0
    for r in relations:
        edge = (r['event1'], r['event2']) if r['relation'] == 'BEFORE' else (r['event2'], r['event1'])
        if (edge[1], edge[0]) in edges:
            violations += 1
        edges.add(edge)
    return violations

def solve_allen_csp(relations, events) -> int:
    """Lightweight Allen algebra CSP solver"""
    if not relations or not events:
        return 0
    
    constraints = defaultdict(bool)
    for r in relations:
        pair = (r['event1'], r['event2']) if r['relation'] == 'BEFORE' else (r['event2'], r['event1'])
        constraints[pair] = True
    
    triggers = list(set(e['trigger'] for e in events))
    n = len(triggers)
    
    if n >= Config.MAX_EVENTS_FOR_CSP:
        import random
        triggers = random.sample(triggers, min(100, n))
        scale = (n / len(triggers)) ** 3
    else:
        scale = 1
    
    violations = 0
    for i, e1 in enumerate(triggers):
        for j, e2 in enumerate(triggers):
            if i == j or (e1, e2) not in constraints:
                continue
            for k, e3 in enumerate(triggers):
                if k in (i, j) and (e2, e3) in constraints and (e3, e1) in constraints:
                    violations += 1
    
    return int(violations * scale)

def extract_constraint_features(relations, events) -> Dict:
    """Extract 3 constraint-based features"""
    if not relations:
        return {'temp_constraint_violation_rate': 0.0, 'temp_constraint_csp_score': 0.0, 'temp_scope_variance': 0.0}
    
    features = {}
    features['temp_constraint_violation_rate'] = safe_division(compute_simple_violations(relations), len(relations))
    features['temp_constraint_csp_score'] = solve_allen_csp(relations, events)
    
    # Scope variance
    positions = {e['trigger']: e.get('trigger_start', i) for i, e in enumerate(events)}
    distances = [abs(positions.get(r['event2'], 0) - positions.get(r['event1'], 0)) for r in relations]
    features['temp_scope_variance'] = float(np.var(distances)) if len(distances) > 1 else 0.0
    
    return features


# Form-meaning alignment


def extract_form_meaning_features(text, events, timex_list, relations, doc_id) -> Dict:
    """Extract 3 form-meaning features (simplified)"""
    features = {}
    text_lower = text.lower()
    
    past_words = ['yesterday', 'ago', 'past', 'previous', 'earlier']
    future_words = ['tomorrow', 'future', 'next', 'later', 'upcoming']
    
    past_tense = sum(1 for e in events if e.get('tense') == 'Past')
    future_tense = sum(1 for e in events if e.get('tense') in ['Fut', 'Pres'])
    past_timex = sum(text_lower.count(w) for w in past_words)
    future_timex = sum(text_lower.count(w) for w in future_words)
    
    tense_ratio = safe_division(past_tense, past_tense + future_tense, 0.5)
    timex_ratio = safe_division(past_timex, past_timex + future_timex, 0.5)
    features['temp_tense_time_alignment'] = 1.0 - abs(tense_ratio - timex_ratio)
    
    deixis_markers = ['now', 'today', 'currently', 'then', 'previously', 'earlier', 'later', 'afterwards', 'recently', 'soon']
    features['temp_deixis_consistency'] = sum(text_lower.count(m) for m in deixis_markers)
    
    sorted_timex = sorted([t for t in timex_list if t.get('position', -1) >= 0], key=lambda x: x['position'])
    shifts, prev_dir = 0, None
    for timex in sorted_timex:
        snippet = timex.get('text', '').lower()
        if any(w in snippet for w in past_words):
            direction = 'past'
        elif any(w in snippet for w in future_words):
            direction = 'future'
        else:
            direction = 'present'
        if prev_dir and prev_dir != direction:
            shifts += 1
        prev_dir = direction
    features['temp_ref_time_shifts'] = shifts
    
    return features


# Graph organization


def extract_graph_organization_features(dag_graph) -> Dict:
    """Extract 3 graph organization features"""
    if dag_graph is None or dag_graph.number_of_nodes() < 2:
        return {'tg_centralization': 0.0, 'tg_clustering_coefficient': 0.0, 'tg_density': 0.0}
    
    features = {}
    
    # Centralization
    try:
        centralities = list(nx.degree_centrality(dag_graph).values())
        max_c = max(centralities)
        n = dag_graph.number_of_nodes()
        features['tg_centralization'] = safe_division(sum(max_c - c for c in centralities), (n-1)*(n-2) if n > 2 else 1)
    except:
        features['tg_centralization'] = 0.0
    
    # Clustering
    try:
        features['tg_clustering_coefficient'] = nx.average_clustering(dag_graph.to_undirected())
    except:
        features['tg_clustering_coefficient'] = 0.0
    
    # Density
    n = dag_graph.number_of_nodes()
    features['tg_density'] = safe_division(dag_graph.number_of_edges(), n*(n-1) if n > 1 else 1)
    
    return features


# Master Extraction


def extract_all_temporal_features(doc_id, events_dict, timex_dict, relations_dict,
                                   raw_graphs, dag_graphs, texts_dict) -> Dict:
    """Extract all 32 temporal features for one document"""
    events = events_dict.get(doc_id, [])
    timex_list = timex_dict.get(doc_id, [])
    relations = relations_dict.get(doc_id, [])
    raw_graph = raw_graphs.get(doc_id)
    dag_graph = dag_graphs.get(doc_id)
    text = texts_dict.get(doc_id, "")
    
    features = {'id': doc_id}
    
    try:
        features.update(extract_event_structure_features(doc_id, events, timex_list, text))
    except:
        features.update({k: 0.0 for k in ['temp_num_events', 'temp_events_per_sentence', 
            'temp_event_lexical_diversity', 'temp_tense_distribution_entropy', 'temp_num_timex', 'temp_timex_event_ratio']})
    
    try:
        features.update(extract_relation_features(relations, raw_graph))
    except:
        features.update({k: 0.0 for k in ['temp_rel_mean_confidence', 'temp_rel_confidence_variance',
            'temp_rel_before_after_ratio', 'temp_rel_raw_cycle_count', 'temp_rel_cycle_edge_ratio',
            'temp_rel_parallel_edge_rate', 'temp_rel_type_entropy', 'temp_rel_transitivity_violation_rate', 'temp_rel_cycle_approx_flag']})
    
    try:
        features.update(extract_graph_features(dag_graph, raw_graph))
    except:
        features.update({k: 0.0 for k in ['tg_edge_retention', 'tg_degree_entropy', 'tg_avg_in_degree',
            'tg_avg_out_degree', 'tg_ordering_entropy', 'tg_longest_path', 'tg_mean_depth', 'tg_branching_factor', 'tg_global_coherence']})
    
    try:
        features.update(extract_constraint_features(relations, events))
    except:
        features.update({k: 0.0 for k in ['temp_constraint_violation_rate', 'temp_constraint_csp_score', 'temp_scope_variance']})
    
    try:
        features.update(extract_form_meaning_features(text, events, timex_list, relations, doc_id))
    except:
        features.update({k: 0.0 for k in ['temp_tense_time_alignment', 'temp_deixis_consistency', 'temp_ref_time_shifts']})
    
    try:
        features.update(extract_graph_organization_features(dag_graph))
    except:
        features.update({k: 0.0 for k in ['tg_centralization', 'tg_clustering_coefficient', 'tg_density']})
    
    return features

def create_empty_features(doc_id) -> Dict:
    """Create zero-filled features for error cases"""
    names = [
        'temp_num_events', 'temp_events_per_sentence', 'temp_event_lexical_diversity',
        'temp_tense_distribution_entropy', 'temp_num_timex', 'temp_timex_event_ratio',
        'temp_rel_mean_confidence', 'temp_rel_confidence_variance', 'temp_rel_before_after_ratio',
        'temp_rel_raw_cycle_count', 'temp_rel_cycle_edge_ratio', 'temp_rel_parallel_edge_rate',
        'temp_rel_type_entropy', 'temp_rel_transitivity_violation_rate', 'temp_rel_cycle_approx_flag',
        'tg_edge_retention', 'tg_degree_entropy', 'tg_avg_in_degree', 'tg_avg_out_degree',
        'tg_ordering_entropy', 'tg_longest_path', 'tg_mean_depth', 'tg_branching_factor', 'tg_global_coherence',
        'temp_constraint_violation_rate', 'temp_constraint_csp_score', 'temp_scope_variance',
        'temp_tense_time_alignment', 'temp_deixis_consistency', 'temp_ref_time_shifts',
        'tg_centralization', 'tg_clustering_coefficient', 'tg_density'
    ]
    return {'id': doc_id, **{n: 0.0 for n in names}}


# Extraction


def run_feature_extraction(events_dict, timex_dict, relations_dict, raw_graphs, 
                           dag_graphs, texts_dict, output_dir: str) -> pd.DataFrame:
    """Run batch feature extraction with checkpointing"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, 'temporal_features.pkl')
    progress_path = os.path.join(output_dir, 'temporal_features_progress.txt')
    
    features_list, start_idx = load_checkpoint(checkpoint_path, progress_path)
    doc_ids = list(events_dict.keys())
    
    print(f"\nExtracting features from {len(doc_ids)} documents (starting from {start_idx})...")
    
    error_count = 0
    start_time = time.time()
    
    for i in tqdm(range(start_idx, len(doc_ids)), desc="Feature extraction"):
        doc_id = doc_ids[i]
        try:
            feats = extract_all_temporal_features(doc_id, events_dict, timex_dict, 
                relations_dict, raw_graphs, dag_graphs, texts_dict)
            features_list.append(feats)
        except Exception as e:
            error_count += 1
            features_list.append(create_empty_features(doc_id))
        
        if (i + 1) % Config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(features_list, checkpoint_path, progress_path, i + 1)
            gc.collect()
    
    # Final save
    df = pd.DataFrame(features_list)
    output_path = os.path.join(output_dir, 'temporal_features_final.csv')
    df.to_csv(output_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"\nExtraction complete: {len(df)} documents in {elapsed/60:.1f} minutes")
    print(f"Errors: {error_count}")
    print(f"Output: {output_path}")
    
    return df


# Validate


def validate_features(df: pd.DataFrame, output_dir: str):
    """Validate extracted features"""
    print("\n" + "="*60)
    print("DATA QUALITY VALIDATION")
    print("="*60)
    
    nan_counts = df.isna().sum()
    inf_counts = df.isin([np.inf, -np.inf]).sum()
    
    print(f"\nNaN values: {nan_counts.drop('id', errors='ignore').sum()}")
    print(f"Inf values: {inf_counts.drop('id', errors='ignore').sum()}")
    
    zero_var = [c for c in df.columns if c != 'id' and df[c].std() == 0]
    print(f"Zero-variance features: {len(zero_var)}")
    if zero_var:
        for c in zero_var:
            print(f"  - {c}")
    
    if 'temp_rel_cycle_approx_flag' in df.columns:
        approx_pct = 100 * (df['temp_rel_cycle_approx_flag'] > 0).mean()
        print(f"Cycle approximation used: {approx_pct:.1f}% of documents")
    
    # Save report
    report_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write("TEMPORAL FEATURES VALIDATION REPORT\n")
        f.write(f"Documents: {len(df)}\n")
        f.write(f"Features: {len(df.columns) - 1}\n\n")
        for col in df.columns:
            if col == 'id':
                continue
            f.write(f"{col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}\n")
    print(f"\nReport saved: {report_path}")


# Main


def main():
    parser = argparse.ArgumentParser(description='Temporal Feature Extraction Pipeline')
    parser.add_argument('--extraction-dir', type=str, required=True, help='Directory with extracted temporal components')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV with text')
    parser.add_argument('--output', type=str, default='./temporal_features_output', help='Output directory')
    args = parser.parse_args()
    
    print("="*60)
    print("TEMPORAL FEATURE EXTRACTION")
    print("="*60)
    
    events, timex, relations, raw_graphs, dag_graphs, dag_type = load_temporal_components(args.extraction_dir)
    texts = load_dataset(args.dataset)
    
    df = run_feature_extraction(events, timex, relations, raw_graphs, dag_graphs, texts, args.output)
    validate_features(df, args.output)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()