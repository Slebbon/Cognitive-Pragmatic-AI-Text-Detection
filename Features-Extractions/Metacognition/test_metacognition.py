"""
Test script for Metacognition Feature Extraction
Validates pipeline with mock data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent to path for import
sys.path.insert(0, str(Path(__file__).parent))
from metacognition_extractor import (
    setup_spacy, load_lexicons, build_all_lookups,
    extract_metacognition_features, process_dataset
)

def create_mock_data() -> pd.DataFrame:
    """Create mock dataset for testing"""
    return pd.DataFrame({
        'id': [f'mock_{i:03d}' for i in range(1, 11)],
        'generation': [
            # High hedging (human-like)
            "It seems that the results might indicate a possible trend. Perhaps we should consider alternative explanations. Some researchers suggest this could be important.",
            # High boosting (AI-like)
            "The results clearly demonstrate that our method is superior. We are absolutely certain this approach is the best. Indeed, the evidence definitely proves our hypothesis.",
            # Mixed markers
            "However, we believe the findings suggest potential applications. In other words, the data shows promising trends. Certainly, further research may be needed.",
            # Self-reference heavy
            "I think we should reconsider our approach. In my opinion, the method needs improvement. We found that our initial assumptions were incorrect.",
            # Evidential heavy
            "According to Smith (2020), the method is effective. Research shows that this approach works. Studies indicate significant improvements.",
            # Weasel words heavy
            "Many experts believe this is true. It is widely known that the effect exists. Some argue that the results are significant.",
            # Reformulation heavy
            "In other words, the method works. That is to say, the results are positive. Namely, we found three key factors.",
            # Minimal markers (neutral)
            "The experiment was conducted. Data was collected. Results were analyzed. The findings were reported.",
            # Complex mixed
            "Although some researchers argue differently, we believe our findings clearly demonstrate the effectiveness. However, it might be necessary to consider limitations.",
            # Short text
            "The results are significant."
        ],
        'is_ai': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1]
    })

def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("METACOGNITION FEATURE EXTRACTION - TEST SUITE")
    print("=" * 70)
    
    # Test 1: Setup
    print("\n[TEST 1] Setting up spaCy...")
    try:
        nlp = setup_spacy()
        print("PASSED: spaCy loaded successfully")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 2: Load lexicons
    print("\n[TEST 2] Loading lexicons...")
    try:
        lexicons = load_lexicons()
        assert len(lexicons) > 0, "No lexicons loaded"
        print(f"PASSED: Loaded {len(lexicons)} lexicons")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 3: Build lookups
    print("\n[TEST 3] Building lookup dictionaries...")
    try:
        lookups = build_all_lookups(lexicons)
        assert len(lookups) > 0, "No lookups built"
        total = sum(len(v) for v in lookups.values())
        print(f"PASSED: Built {len(lookups)} lookups with {total} total markers")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 4: Single document extraction
    print("\n[TEST 4] Single document feature extraction...")
    try:
        test_text = "However, we believe the results clearly show that the method works."
        doc = nlp(test_text)
        features = extract_metacognition_features(doc, "single_test", lookups)
        assert features['id'] == "single_test"
        assert features['doc_length_tokens'] > 0
        assert 'hedges_density' in features
        assert 'boosters_density' in features
        print(f"PASSED: Extracted {len(features)} features")
        print(f"  - doc_length_tokens: {features['doc_length_tokens']}")
        print(f"  - hedges_density: {features['hedges_density']:.4f}")
        print(f"  - boosters_density: {features['boosters_density']:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 5: Batch processing
    print("\n[TEST 5] Batch processing mock data...")
    try:
        mock_df = create_mock_data()
        features_df = process_dataset(mock_df, nlp, lookups, 'generation', batch_size=4)
        assert len(features_df) == len(mock_df), "Row count mismatch"
        assert 'id' in features_df.columns
        print(f"PASSED: Processed {len(features_df)} documents")
        print(f"  - Output shape: {features_df.shape}")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 6: Feature value validation
    print("\n[TEST 6] Validating feature values...")
    try:
        # Check no NaN in critical columns
        critical_cols = ['id', 'doc_length_tokens', 'hedges_density', 'boosters_density']
        for col in critical_cols:
            nan_count = features_df[col].isna().sum()
            assert nan_count == 0, f"NaN found in {col}"
        
        # Check density values are non-negative
        density_cols = [c for c in features_df.columns if 'density' in c]
        for col in density_cols:
            assert (features_df[col] >= 0).all(), f"Negative values in {col}"
        
        print(f"PASSED: All {len(density_cols)} density features are valid")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Test 7: Compare AI vs Human patterns
    print("\n[TEST 7] Checking AI vs Human patterns...")
    try:
        merged = features_df.merge(mock_df[['id', 'is_ai']], on='id')
        ai_hedges = merged[merged['is_ai'] == 1]['hedges_density'].mean()
        human_hedges = merged[merged['is_ai'] == 0]['hedges_density'].mean()
        ai_boosters = merged[merged['is_ai'] == 1]['boosters_density'].mean()
        human_boosters = merged[merged['is_ai'] == 0]['boosters_density'].mean()
        
        print(f"PASSED: Pattern analysis complete")
        print(f"  - AI hedges density: {ai_hedges:.4f}")
        print(f"  - Human hedges density: {human_hedges:.4f}")
        print(f"  - AI boosters density: {ai_boosters:.4f}")
        print(f"  - Human boosters density: {human_boosters:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    
    # Save test output
    print("\n[TEST 8] Saving test output...")
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
    print("\nFeature columns extracted:")
    for i, col in enumerate(features_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)