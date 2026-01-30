# -*- coding: utf-8 -*-
"""
Test script to verify coreference feature extraction setup
"""

import sys

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    errors = []
    
    try:
        import torch
        print(f"  torch: OK ({torch.__version__})")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        errors.append(f"torch: {e}")
    
    try:
        import spacy
        print(f"  spacy: OK ({spacy.__version__})")
    except ImportError as e:
        errors.append(f"spacy: {e}")
    
    try:
        from fastcoref import FCoref
        print(f"  fastcoref: OK")
    except ImportError as e:
        errors.append(f"fastcoref: {e}")
    
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
        from tqdm import tqdm
        print(f"  tqdm: OK")
    except ImportError as e:
        errors.append(f"tqdm: {e}")
    
    return errors

def test_spacy_model():
    """Test spaCy model loading"""
    print("\nTesting spaCy model...")
    import spacy
    
    try:
        nlp = spacy.load("en_core_web_trf", disable=["lemmatizer"])
        doc = nlp("John went to the store. He bought some milk.")
        print(f"  Model loaded, found {len(list(doc.ents))} entities")
        return nlp
    except OSError:
        print("  Model not found. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
        nlp = spacy.load("en_core_web_trf", disable=["lemmatizer"])
        return nlp

def test_fastcoref(device='cpu'):
    """Test FastCoref model"""
    print("\nTesting FastCoref...")
    from fastcoref import FCoref
    
    try:
        coref = FCoref(device=device)
        text = "John went to the store. He bought some milk. Then he went home."
        preds = coref.predict(texts=[text])
        clusters = preds[0].get_clusters(as_strings=True)
        print(f"  Model loaded, found {len(clusters)} coreference chains")
        print(f"  Sample chain: {clusters[0] if clusters else 'None'}")
        return coref
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def test_feature_extraction(coref_model, nlp):
    """Test feature extraction on sample text"""
    print("\nTesting feature extraction...")
    
    from coreference_extraction_local import extract_all_features
    
    sample_text = """
    Dr. Sarah Johnson arrived at the hospital early that morning. She had been working 
    on a complex case for weeks. The patient, Mr. Williams, had shown significant improvement 
    since his surgery. Dr. Johnson reviewed his charts carefully. She noted that the treatment 
    was progressing well. Mr. Williams thanked her for her dedication to his care.
    """
    
    try:
        features = extract_all_features(sample_text, coref_model, nlp)
        non_zero = sum(1 for k, v in features.items() if isinstance(v, (int, float)) and v > 0)
        print(f"  Extracted {len(features)} features, {non_zero} non-zero")
        
        print("\n  Sample features:")
        for k, v in list(features.items())[:10]:
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_tiers(coref_model, nlp):
    """Test each feature tier individually"""
    print("\nTesting individual feature tiers...")
    
    from coreference_extraction_local import (
        extract_chains_from_fastcoref,
        calculate_pronoun_ratio,
        calculate_tier1_rmo_features,
        calculate_tier2_mta_features,
        calculate_supplementary_pronoun_features,
        calculate_tier3_context_features,
        calculate_tier4_features
    )
    
    text = "The company announced its new product. It will be available next month. The CEO said he was excited."
    
    chains, all_mentions, doc = extract_chains_from_fastcoref(text, coref_model, nlp)
    print(f"  Chains extracted: {len(chains)}, Mentions: {len(all_mentions)}")
    
    tiers = [
        ("Baseline", lambda: calculate_pronoun_ratio(all_mentions)),
        ("Tier 1 RMO", lambda: calculate_tier1_rmo_features(chains, doc)),
        ("Tier 2 MTA", lambda: calculate_tier2_mta_features(chains, all_mentions, doc)),
        ("Supp Pronoun", lambda: calculate_supplementary_pronoun_features(chains)),
        ("Tier 3 Context", lambda: calculate_tier3_context_features(chains, doc)),
        ("Tier 4 Min+Sing", lambda: calculate_tier4_features(chains, doc))
    ]
    
    all_ok = True
    for name, func in tiers:
        try:
            result = func()
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("COREFERENCE PIPELINE TEST SUITE")
    print("="*60)
    
    # Test imports
    errors = test_imports()
    if errors:
        print("\nMissing dependencies:")
        for e in errors:
            print(f"  - {e}")
        print("\nInstall with:")
        print("  pip install torch spacy fastcoref pandas numpy tqdm transformers")
        print("  python -m spacy download en_core_web_trf")
        sys.exit(1)
    
    # Determine device
    import torch
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Test spaCy
    nlp = test_spacy_model()
    if nlp is None:
        print("\nspaCy model test FAILED")
        sys.exit(1)
    
    # Test FastCoref
    coref_model = test_fastcoref(device)
    if coref_model is None:
        print("\nFastCoref test FAILED")
        sys.exit(1)
    
    # Test feature extraction
    if not test_feature_extraction(coref_model, nlp):
        print("\nFeature extraction test FAILED")
        sys.exit(1)
    
    # Test all tiers
    if not test_all_tiers(coref_model, nlp):
        print("\nSome tier tests FAILED")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
    print("\nThe pipeline is ready to use.")
    print("\nUsage:")
    print("  python coreference_extraction_local.py \\")
    print("    --dataset /path/to/dataset.csv \\")
    print("    --text-column generation \\")
    print("    --output ./coreference_output")

if __name__ == '__main__':
    main()