# -*- coding: utf-8 -*-
"""
Quick test script to verify temporal extraction pipeline setup
Run this before the full pipeline to check dependencies
"""

import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    errors = []
    
    try:
        import torch
        print(f"  torch: OK (version {torch.__version__})")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        errors.append(f"torch: {e}")
    
    try:
        import stanza
        print(f"  stanza: OK")
    except ImportError as e:
        errors.append(f"stanza: {e}")
    
    try:
        import transformers
        print(f"  transformers: OK (version {transformers.__version__})")
    except ImportError as e:
        errors.append(f"transformers: {e}")
    
    try:
        import networkx as nx
        print(f"  networkx: OK (version {nx.__version__})")
    except ImportError as e:
        errors.append(f"networkx: {e}")
    
    try:
        import pandas as pd
        print(f"  pandas: OK (version {pd.__version__})")
    except ImportError as e:
        errors.append(f"pandas: {e}")
    
    try:
        import numpy as np
        print(f"  numpy: OK (version {np.__version__})")
    except ImportError as e:
        errors.append(f"numpy: {e}")
    
    try:
        from tqdm import tqdm
        print(f"  tqdm: OK")
    except ImportError as e:
        errors.append(f"tqdm: {e}")
    
    try:
        from stanfordcorenlp import StanfordCoreNLP
        print(f"  stanfordcorenlp: OK")
    except ImportError as e:
        errors.append(f"stanfordcorenlp: {e} (optional, needed for SUTime)")
    
    return errors

def test_stanza_model():
    """Test Stanza model loading"""
    print("\nTesting Stanza model...")
    import stanza
    import torch
    
    try:
        stanza.download('en', verbose=False)
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                             use_gpu=torch.cuda.is_available(), verbose=False)
        
        doc = nlp("The meeting happened yesterday and lasted two hours.")
        verbs = [w.text for s in doc.sentences for w in s.words if w.upos == 'VERB']
        print(f"  Model loaded, found verbs: {verbs}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_sample_graph():
    """Test graph construction logic"""
    print("\nTesting graph construction...")
    import networkx as nx
    
    G = nx.DiGraph()
    G.add_edge("started", "ended", relation="BEFORE", confidence=0.95)
    G.add_edge("ended", "reported", relation="BEFORE", confidence=0.87)
    
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"  Created graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Is DAG: {is_dag}")
    return is_dag

def main():
    print("TEMPORAL PIPELINE TEST")
    
    errors = test_imports()
    
    if errors:
        print("\nMissing dependencies:")
        for e in errors:
            print(f"  - {e}")
        print("\nInstall with: pip install torch stanza transformers networkx pandas numpy tqdm stanfordcorenlp")
    
    stanza_ok = test_stanza_model()
    graph_ok = test_sample_graph()
    
    if not errors and stanza_ok and graph_ok:
        print("PASSATO")
    else:
        print("FALLITO")
        sys.exit(1)

if __name__ == '__main__':
    main()