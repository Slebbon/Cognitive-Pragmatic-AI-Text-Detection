# -*- coding: utf-8 -*-
"""
Test script to verify temporal feature extraction setup
Creates synthetic data and validates all feature categories
"""

import sys
import tempfile
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    errors = []
    
    try:
        import pandas as pd
        print(f"  pandas: OK ({pd.__version__})")
    except ImportError as e:
        errors.append(f"pandas: {e}")
    
    try:
        import numpy as np
        print(f"  numpy: OK ({np.__version__})")
    except ImportError as e:
        errors.append(f"numpy: {e}")
    
    try:
        import networkx as nx
        print(f"  networkx: OK ({nx.__version__})")
    except ImportError as e:
        errors.append(f"networkx: {e}")
    
    try:
        from scipy.stats import entropy
        print(f"  scipy: OK")
    except ImportError as e:
        errors.append(f"scipy: {e}")
    
    try:
        from tqdm import tqdm
        print(f"  tqdm: OK")
    except ImportError as e:
        errors.append(f"tqdm: {e}")
    
    return errors

def create_synthetic_data():
    """Create synthetic temporal components for testing"""
    print("\nCreating synthetic test data...")
    
    # Synthetic events
    events_dict = {
        'doc_001': [
            {'trigger': 'started', 'lemma': 'start', 'tense': 'Past', 'trigger_start': 10},
            {'trigger': 'continued', 'lemma': 'continue', 'tense': 'Past', 'trigger_start': 30},
            {'trigger': 'finished', 'lemma': 'finish', 'tense': 'Past', 'trigger_start': 50},
        ],
        'doc_002': [
            {'trigger': 'will begin', 'lemma': 'begin', 'tense': 'Fut', 'trigger_start': 5},
            {'trigger': 'expects', 'lemma': 'expect', 'tense': 'Pres', 'trigger_start': 25},
        ],
        'doc_003': []
    }
    
    # Synthetic TIMEX
    timex_dict = {
        'doc_001': [
            {'text': 'yesterday', 'type': 'DATE', 'position': 0},
            {'text': '3pm', 'type': 'TIME', 'position': 20},
        ],
        'doc_002': [{'text': 'tomorrow', 'type': 'DATE', 'position': 0}],
        'doc_003': []
    }
    
    # Synthetic relations
    relations_dict = {
        'doc_001': [
            {'event1': 'started', 'event2': 'continued', 'relation': 'BEFORE', 'confidence': 0.95},
            {'event1': 'continued', 'event2': 'finished', 'relation': 'BEFORE', 'confidence': 0.88},
            {'event1': 'started', 'event2': 'finished', 'relation': 'BEFORE', 'confidence': 0.92},
        ],
        'doc_002': [
            {'event1': 'will begin', 'event2': 'expects', 'relation': 'AFTER', 'confidence': 0.75},
        ],
        'doc_003': []
    }
    
    # Build graphs
    import networkx as nx
    
    raw_graphs = {}
    dag_graphs = {}
    
    for doc_id, rels in relations_dict.items():
        if rels:
            G = nx.DiGraph()
            for r in rels:
                if r['relation'] == 'BEFORE':
                    G.add_edge(r['event1'], r['event2'], confidence=r['confidence'])
                else:
                    G.add_edge(r['event2'], r['event1'], confidence=r['confidence'])
            raw_graphs[doc_id] = G
            dag_graphs[doc_id] = G.copy()
            dag_graphs[doc_id].graph['edge_retention_rate'] = 1.0
        else:
            raw_graphs[doc_id] = None
            dag_graphs[doc_id] = None
    
    # Synthetic texts
    texts_dict = {
        'doc_001': "Yesterday, the meeting started at noon. It continued for two hours and finished at 3pm.",
        'doc_002': "Tomorrow the project will begin. Everyone expects great results.",
        'doc_003': "This is a simple text without temporal content."
    }
    
    print(f"  Created {len(events_dict)} synthetic documents")
    return events_dict, timex_dict, relations_dict, raw_graphs, dag_graphs, texts_dict

def test_feature_extraction():
    """Test all feature extraction functions"""
    print("\nTesting feature extraction functions...")
    
    from temporal_features_local import (
        extract_event_structure_features,
        extract_relation_features,
        extract_graph_features,
        extract_constraint_features,
        extract_form_meaning_features,
        extract_graph_organization_features,
        extract_all_temporal_features
    )
    
    events_dict, timex_dict, relations_dict, raw_graphs, dag_graphs, texts_dict = create_synthetic_data()
    
    doc_id = 'doc_001'
    events = events_dict[doc_id]
    timex = timex_dict[doc_id]
    relations = relations_dict[doc_id]
    raw_graph = raw_graphs[doc_id]
    dag_graph = dag_graphs[doc_id]
    text = texts_dict[doc_id]
    
    # Test Category I
    print("\n  Category I: Event Structure...")
    try:
        cat1 = extract_event_structure_features(doc_id, events, timex, text)
        assert 'temp_num_events' in cat1
        assert cat1['temp_num_events'] == 3
        print(f"    OK - {len(cat1)} features extracted")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    # Test Category II
    print("  Category II: Relation-Level...")
    try:
        cat2 = extract_relation_features(relations, raw_graph)
        assert 'temp_rel_mean_confidence' in cat2
        print(f"    OK - {len(cat2)} features extracted")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    # Test Category III
    print("  Category III: Graph-Theoretic...")
    try:
        cat3 = extract_graph_features(dag_graph, raw_graph)
        assert 'tg_edge_retention' in cat3
        print(f"    OK - {len(cat3)} features extracted")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    # Test Category IV
    print("  Category IV: Constraint-Based...")
    try:
        cat4 = extract_constraint_features(relations, events)
        assert 'temp_constraint_violation_rate' in cat4
        print(f"    OK - {len(cat4)} features extracted")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    # Test Category V
    print("  Category V: Form-Meaning...")
    try:
        cat5 = extract_form_meaning_features(text, events, timex, relations, doc_id)
        assert 'temp_tense_time_alignment' in cat5
        print(f"    OK - {len(cat5)} features extracted")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    # Test Category VI
    print("  Category VI: Graph Organization...")
    try:
        cat6 = extract_graph_organization_features(dag_graph)
        assert 'tg_centralization' in cat6
        print(f"    OK - {len(cat6)} features extracted")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    # Test master function
    print("\n  Testing master extraction function...")
    try:
        all_feats = extract_all_temporal_features(
            doc_id, events_dict, timex_dict, relations_dict,
            raw_graphs, dag_graphs, texts_dict
        )
        n_features = len(all_feats) - 1  # exclude 'id'
        print(f"    OK - {n_features} total features extracted")
        
        # Verify expected count
        expected = 6 + 9 + 9 + 3 + 3 + 3  # 33 features
        if n_features >= 30:
            print(f"    Feature count: {n_features} (expected ~{expected})")
        else:
            print(f"    WARNING: Only {n_features} features (expected ~{expected})")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    return True

def test_edge_cases():
    """Test handling of edge cases"""
    print("\nTesting edge cases...")
    
    from temporal_features_local import extract_all_temporal_features, create_empty_features
    
    # Empty document
    print("  Empty document...")
    try:
        feats = extract_all_temporal_features(
            'empty_doc',
            {'empty_doc': []},
            {'empty_doc': []},
            {'empty_doc': []},
            {'empty_doc': None},
            {'empty_doc': None},
            {'empty_doc': ""}
        )
        assert feats['temp_num_events'] == 0
        print("    OK - handled gracefully")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    # Error fallback
    print("  Error fallback function...")
    try:
        empty = create_empty_features('error_doc')
        assert 'id' in empty
        assert empty['temp_num_events'] == 0.0
        print("    OK - returns zero-filled dict")
    except Exception as e:
        print(f"    FAILED: {e}")
        return False
    
    return True

def test_batch_processing():
    """Test batch processing simulation"""
    print("\nTesting batch processing...")
    
    import pandas as pd
    from temporal_features_local import extract_all_temporal_features
    
    events_dict, timex_dict, relations_dict, raw_graphs, dag_graphs, texts_dict = create_synthetic_data()
    
    features_list = []
    for doc_id in events_dict.keys():
        feats = extract_all_temporal_features(
            doc_id, events_dict, timex_dict, relations_dict,
            raw_graphs, dag_graphs, texts_dict
        )
        features_list.append(feats)
    
    df = pd.DataFrame(features_list)
    print(f"  Created DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Check for NaN/Inf
    nan_count = df.isna().sum().sum()
    inf_count = df.isin([float('inf'), float('-inf')]).sum().sum()
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    
    if nan_count == 0 and inf_count == 0:
        print("  OK - No invalid values")
        return True
    else:
        print("  WARNING - Found invalid values")
        return False

def main():
    print("="*60)
    print("TEMPORAL FEATURES TEST SUITE")
    print("="*60)
    
    # Test imports
    errors = test_imports()
    if errors:
        print("\nMissing dependencies:")
        for e in errors:
            print(f"  - {e}")
        print("\nInstall with: pip install pandas numpy networkx scipy tqdm")
        sys.exit(1)
    
    # Test feature extraction
    if not test_feature_extraction():
        print("\nFeature extraction tests FAILED")
        sys.exit(1)
    
    # Test edge cases
    if not test_edge_cases():
        print("\nEdge case tests FAILED")
        sys.exit(1)
    
    # Test batch processing
    if not test_batch_processing():
        print("\nBatch processing tests FAILED")
        sys.exit(1)
    
 
if __name__ == '__main__':
    main()