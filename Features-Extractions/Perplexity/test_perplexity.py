"""
Test script for Perplexity Feature Extraction
Validates pipeline with mock data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

def create_mock_data() -> pd.DataFrame:
    """Create mock dataset for testing"""
    return pd.DataFrame({
        'id': [f'mock_{i:03d}' for i in range(1, 11)],
        'generation': [
            # Fluent, predictable text (low perplexity expected)
            "The cat sat on the mat. The dog ran in the park. Birds fly in the sky. Fish swim in the sea.",
            # Complex academic text (higher perplexity)
            "The epistemological ramifications of quantum entanglement necessitate a paradigmatic reconceptualization of causality within the framework of relativistic spacetime.",
            # Simple narrative
            "John went to the store. He bought some milk and bread. Then he walked home. It was a nice day outside.",
            # Technical jargon
            "The API endpoint returns a JSON payload containing the serialized object graph with lazy-loaded associations materialized via proxy instantiation.",
            # Casual conversational
            "Hey what's up! I was thinking we could grab lunch later. There's this new place downtown that everyone's been talking about.",
            # Repetitive structure (very predictable)
            "I like apples. I like oranges. I like bananas. I like grapes. I like mangoes. I like peaches.",
            # Mixed complexity
            "While the fundamental principles remain unchanged, recent developments have significantly altered our understanding of the underlying mechanisms.",
            # Short simple
            "The sun is bright. Water is wet. Fire is hot.",
            # Long complex sentence
            "Despite the considerable challenges posed by the unprecedented global circumstances, the organization managed to exceed its quarterly targets through innovative strategic pivots and enhanced digital engagement.",
            # Creative writing
            "The ancient forest whispered secrets to those who listened. Shadows danced between the towering oaks as twilight painted the sky in hues of amber and violet."
        ],
        'is_ai': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    })

def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("PERPLEXITY FEATURE EXTRACTION - TEST SUITE")
    print("=" * 70)
    
    # Test 1: Import modules
    print("\n[TEST 1] Importing modules...")
    try:
        from perplexity_extractor import (
            setup_device, load_language_model, load_spacy,
            compute_log_perplexity, extract_perplexity_features,
            create_perturbation
        )
        print("PASSED: All modules imported")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 2: Setup device
    print("\n[TEST 2] Setting up device...")
    try:
        device = setup_device()
        print(f"PASSED: Device = {device}")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 3: Load model
    print("\n[TEST 3] Loading language model...")
    try:
        model, tokenizer = load_language_model(device)
        print("PASSED: Model loaded")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 4: Load spaCy
    print("\n[TEST 4] Loading spaCy...")
    try:
        nlp = load_spacy()
        print("PASSED: spaCy loaded")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 5: Single perplexity computation
    print("\n[TEST 5] Computing single perplexity...")
    try:
        test_text = "The quick brown fox jumps over the lazy dog."
        ppl = compute_log_perplexity(test_text, model, tokenizer, device)
        assert not np.isnan(ppl), "Perplexity is NaN"
        assert ppl > 0, "Perplexity should be positive"
        print(f"PASSED: Perplexity = {ppl:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 6: Perturbation creation
    print("\n[TEST 6] Creating perturbations...")
    try:
        original = "This is a test sentence with several words to perturb."
        perturbed = create_perturbation(original)
        assert original != perturbed, "Perturbation should modify text"
        assert len(original.split()) == len(perturbed.split()), "Word count should match"
        print(f"PASSED: Original vs perturbed differ")
        print(f"  Original:  {original[:50]}...")
        print(f"  Perturbed: {perturbed[:50]}...")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 7: Full feature extraction
    print("\n[TEST 7] Full feature extraction on single document...")
    try:
        test_text = "The cat sat on the mat. The dog ran in the park. Birds fly in the sky."
        doc = nlp(test_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        features = extract_perplexity_features(test_text, sentences, "test_001", model, tokenizer, device)
        
        assert features['id'] == "test_001"
        assert 'doc_perplexity' in features
        assert 'perturbation_discrepancy' in features
        
        print("PASSED: All 8 features extracted")
        for key, val in features.items():
            if key != 'id':
                print(f"  {key}: {val:.4f}" if not np.isnan(val) else f"  {key}: NaN")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 8: Batch processing mock data
    print("\n[TEST 8] Batch processing mock data...")
    try:
        from perplexity_extractor import process_dataset
        mock_df = create_mock_data()
        
        # Process only first 3 for speed
        small_df = mock_df.head(3)
        features_df = process_dataset(small_df, 'generation', model, tokenizer, nlp, device)
        
        assert len(features_df) == 3, "Should have 3 rows"
        assert 'doc_perplexity' in features_df.columns
        print(f"PASSED: Processed {len(features_df)} documents")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 9: Feature validity
    print("\n[TEST 9] Validating feature values...")
    try:
        # Check perplexity is reasonable (typically 1-100 for log-perplexity)
        valid_ppl = features_df['doc_perplexity'].dropna()
        assert len(valid_ppl) > 0, "No valid perplexity values"
        assert (valid_ppl > 0).all(), "Perplexity should be positive"
        assert (valid_ppl < 100).all(), "Perplexity unusually high"
        
        print(f"PASSED: Perplexity range [{valid_ppl.min():.2f}, {valid_ppl.max():.2f}]")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 10: Save output
    print("\n[TEST 10] Saving test output...")
    try:
        output_path = Path(__file__).parent / "output" / "test_output.csv"
        output_path.parent.mkdir(exist_ok=True)
        features_df.to_csv(output_path, index=False)
        print(f"PASSED: Saved to {output_path}")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    
    print("\nFeatures extracted:")
    for i, col in enumerate(features_df.columns, 1):
        print(f"  {i}. {col}")
    
    print("\nSample output:")
    print(features_df[['id', 'doc_perplexity', 'mean_sentence_perplexity', 'perturbation_discrepancy']].to_string())
    
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)