"""
Perplexity Feature Extraction Pipeline - Local Version
Extracts 8 perplexity-based features for AI text detection using Pythia-160M
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import argparse
import warnings

warnings.filterwarnings('ignore')


#Configurazione

BASE_PATH = Path(__file__).parent
OUTPUT_PATH = BASE_PATH / "output"
CHECKPOINT_PATH = BASE_PATH / "checkpoints"
OUTPUT_PATH.mkdir(exist_ok=True)
CHECKPOINT_PATH.mkdir(exist_ok=True)

MODEL_NAME = 'EleutherAI/pythia-160m'
MAX_LENGTH = 512
NUM_PERTURBATIONS = 5
CHECKPOINT_INTERVAL = 100


# GPU AND MODEL SETUP

def setup_device():
    """Initialize device (GPU if available)"""
    print("=" * 60)
    print("SETTING UP ENVIRONMENT")
    print("=" * 60)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("Using CPU (will be slower)")
    
    return device

def load_language_model(device):
    """Load Pythia language model for perplexity computation"""
    print(f"\nLoading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {allocated:.2f}/{total:.2f} GB")
    
    return model, tokenizer

def load_spacy():
    """Load spaCy for sentence segmentation"""
    print("\nLoading spaCy...")
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print("Downloading en_core_web_lg...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"], check=True)
        nlp = spacy.load("en_core_web_lg")
    
    nlp.disable_pipes(['ner', 'lemmatizer'])
    print("spaCy loaded")
    return nlp


#Perplexity computations

def compute_log_perplexity(text: str, model, tokenizer, device, max_length: int = 512) -> float:
    """Compute log-perplexity (loss) for a text segment"""
    try:
        if len(text.split()) < 3:
            return np.nan
        
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding=False)
        input_ids = encodings.input_ids.to(device)
        
        if input_ids.shape[1] < 2:
            return np.nan
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        
        return loss.item()
    except:
        return np.nan

def compute_token_log_probabilities(text: str, model, tokenizer, device, max_length: int = 512) -> np.ndarray:
    """Compute per-token log probabilities for entropy calculation"""
    try:
        if len(text.split()) < 3:
            return np.array([])
        
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length, padding=False)
        input_ids = encodings.input_ids.to(device)
        
        if input_ids.shape[1] < 2:
            return np.array([])
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        return token_log_probs.cpu().numpy().flatten()
    except:
        return np.array([])

def create_perturbation(text: str) -> str:
    """Create orthographic perturbation by swapping characters in ~15% of words"""
    words = text.split()
    num_words = len(words)
    num_to_perturb = max(1, int(0.15 * num_words))
    perturb_positions = np.random.choice(num_words, num_to_perturb, replace=False)
    
    perturbed_words = []
    for j, word in enumerate(words):
        if j in perturb_positions and len(word) > 3:
            pos = np.random.randint(0, len(word) - 1)
            word_list = list(word)
            word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
            perturbed_words.append(''.join(word_list))
        else:
            perturbed_words.append(word)
    
    return ' '.join(perturbed_words)


# FEATURE EXTRACTION

def create_empty_features(doc_id: str) -> Dict:
    """Create zero-filled feature dict for error cases"""
    return {
        'id': doc_id,
        'doc_perplexity': np.nan,
        'mean_sentence_perplexity': np.nan,
        'sentence_perplexity_variance': np.nan,
        'token_probability_entropy': np.nan,
        'perplexity_curvature': np.nan,
        'perplexity_burstiness': np.nan,
        'perplexity_trajectory_slope': np.nan,
        'perturbation_discrepancy': np.nan
    }

def extract_perplexity_features(text: str, sentences: List[str], doc_id: str, 
                                 model, tokenizer, device) -> Dict:
    """Extract all 8 perplexity features from a document"""
    try:
        features = {'id': doc_id}
        
        # Feature 1: Document-level perplexity
        doc_ppl = compute_log_perplexity(text, model, tokenizer, device, MAX_LENGTH)
        features['doc_perplexity'] = doc_ppl
        
        # Feature 2-3: Sentence-level perplexity stats
        sent_ppls = []
        if len(sentences) > 0:
            for sent in sentences:
                sent_ppl = compute_log_perplexity(sent, model, tokenizer, device, MAX_LENGTH)
                if not np.isnan(sent_ppl) and np.isfinite(sent_ppl):
                    sent_ppls.append(sent_ppl)
            
            features['mean_sentence_perplexity'] = np.mean(sent_ppls) if sent_ppls else np.nan
            features['sentence_perplexity_variance'] = np.var(sent_ppls) if len(sent_ppls) > 1 else np.nan
        else:
            features['mean_sentence_perplexity'] = np.nan
            features['sentence_perplexity_variance'] = np.nan
        
        # Feature 4: Token probability entropy
        token_log_probs = compute_token_log_probabilities(text, model, tokenizer, device, MAX_LENGTH)
        if len(token_log_probs) > 0:
            token_probs = np.exp(token_log_probs)
            features['token_probability_entropy'] = -np.sum(token_probs * token_log_probs)
        else:
            features['token_probability_entropy'] = np.nan
        
        # Feature 5: Perplexity curvature (doc vs sentence)
        if not np.isnan(doc_ppl) and not np.isnan(features['mean_sentence_perplexity']):
            features['perplexity_curvature'] = doc_ppl - features['mean_sentence_perplexity']
        else:
            features['perplexity_curvature'] = np.nan
        
        # Feature 6: Perplexity burstiness (max/min ratio)
        if len(sent_ppls) > 1:
            valid_ppls = [p for p in sent_ppls if np.isfinite(p) and p > 0]
            if len(valid_ppls) > 1:
                features['perplexity_burstiness'] = np.max(valid_ppls) / np.min(valid_ppls)
            else:
                features['perplexity_burstiness'] = np.nan
        else:
            features['perplexity_burstiness'] = np.nan
        
        # Feature 7: Perplexity trajectory slope
        if len(sent_ppls) > 2:
            valid_indices = [j for j, p in enumerate(sent_ppls) if np.isfinite(p)]
            if len(valid_indices) > 2:
                valid_ppls = [sent_ppls[j] for j in valid_indices]
                try:
                    slope, _ = np.polyfit(valid_indices, valid_ppls, 1)
                    features['perplexity_trajectory_slope'] = slope
                except:
                    features['perplexity_trajectory_slope'] = np.nan
            else:
                features['perplexity_trajectory_slope'] = np.nan
        else:
            features['perplexity_trajectory_slope'] = np.nan
        
        # Feature 8: Perturbation discrepancy
        if not np.isnan(doc_ppl) and len(text.split()) > 20:
            perturbed_ppls = []
            for _ in range(NUM_PERTURBATIONS):
                try:
                    perturbed_text = create_perturbation(text)
                    pert_ppl = compute_log_perplexity(perturbed_text, model, tokenizer, device, MAX_LENGTH)
                    if not np.isnan(pert_ppl) and np.isfinite(pert_ppl):
                        perturbed_ppls.append(pert_ppl)
                except:
                    continue
            
            if len(perturbed_ppls) >= 2:
                mean_pert = np.mean(perturbed_ppls)
                std_pert = np.std(perturbed_ppls)
                if std_pert > 0:
                    features['perturbation_discrepancy'] = (doc_ppl - mean_pert) / std_pert
                else:
                    features['perturbation_discrepancy'] = doc_ppl - mean_pert
            else:
                features['perturbation_discrepancy'] = np.nan
        else:
            features['perturbation_discrepancy'] = np.nan
        
        return features
    
    except Exception as e:
        print(f"Error processing {doc_id}: {e}")
        return create_empty_features(doc_id)


#Segmentazione 

def segment_sentences(texts_dict: Dict[str, str], nlp) -> Dict[str, List[str]]:
    """Segment all texts into sentences"""
    print("\n" + "=" * 60)
    print("SEGMENTING SENTENCES")
    print("=" * 60)
    
    segmentation = {}
    for doc_id, text in tqdm(texts_dict.items(), desc="Segmenting"):
        try:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            segmentation[doc_id] = sentences
        except:
            segmentation[doc_id] = []
    
    sent_counts = [len(s) for s in segmentation.values()]
    print(f"Mean sentences/doc: {np.mean(sent_counts):.1f}")
    return segmentation


#Processing batch + checkpoint

def process_dataset(df: pd.DataFrame, text_col: str, model, tokenizer, nlp, device) -> pd.DataFrame:
    """Process entire dataset and extract features"""
    print("\n" + "=" * 60)
    print(f"PROCESSING {len(df)} DOCUMENTS")
    print("=" * 60)
    
    # Create texts dict
    texts_dict = {row['id']: row[text_col] for _, row in df.iterrows()}
    
    # Segment sentences
    segmentation = segment_sentences(texts_dict, nlp)
    
    # Extract features
    print("\nExtracting perplexity features...")
    features_list = []
    doc_ids = list(texts_dict.keys())
    
    checkpoint_file = CHECKPOINT_PATH / "perplexity_checkpoint.pkl"
    progress_file = CHECKPOINT_PATH / "perplexity_progress.txt"
    
    # Check for checkpoint
    start_idx = 0
    if checkpoint_file.exists() and progress_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                features_list = pickle.load(f)
            with open(progress_file, 'r') as f:
                start_idx = int(f.read().strip())
            print(f"Resuming from document {start_idx}")
        except:
            start_idx = 0
            features_list = []
    
    for i in tqdm(range(start_idx, len(doc_ids)), desc="Computing perplexity"):
        doc_id = doc_ids[i]
        text = texts_dict[doc_id]
        sentences = segmentation[doc_id]
        
        features = extract_perplexity_features(text, sentences, doc_id, model, tokenizer, device)
        features_list.append(features)
        
        # Clear GPU cache
        if torch.cuda.is_available() and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        
        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(features_list, f)
            with open(progress_file, 'w') as f:
                f.write(str(i + 1))
            print(f"\nCheckpoint saved at {i + 1}")
    
    return pd.DataFrame(features_list)


#Main

def main():
    parser = argparse.ArgumentParser(description='Extract perplexity features from text')
    parser.add_argument('--input', type=str, help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Path to output CSV file')
    parser.add_argument('--text_col', type=str, default='generation', help='Column name containing text')
    parser.add_argument('--test', action='store_true', help='Run test with mock data')
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    model, tokenizer = load_language_model(device)
    nlp = load_spacy()
    
    if args.test:
        print("\n" + "=" * 60)
        print("RUNNING TEST WITH MOCK DATA")
        print("=" * 60)
        
        from test_perplexity import create_mock_data
        mock_df = create_mock_data()
        
        features_df = process_dataset(mock_df, 'generation', model, tokenizer, nlp, device)
        output_path = OUTPUT_PATH / "test_perplexity_features.csv"
        features_df.to_csv(output_path, index=False)
        
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Shape: {features_df.shape}")
        print(f"\nSample features:")
        print(features_df[['id', 'doc_perplexity', 'perturbation_discrepancy']].head())
        print(f"\nOutput saved to: {output_path}")
        print("\nTEST PASSED!")
        return
    
    # Interactive input
    if args.input is None:
        print("\n" + "=" * 60)
        print("INPUT FILE SELECTION")
        print("=" * 60)
        args.input = input("Enter path to input CSV file: ").strip()
    
    # Load data
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
            print(f"ERROR: Text column not found. Available: {df.columns.tolist()}")
            return
    
    print(f"Using text column: '{text_col}'")
    print(f"Dataset: {len(df)} documents")
    
    # Process
    features_df = process_dataset(df, text_col, model, tokenizer, nlp, device)
    
    # Save
    if args.output is None:
        args.output = OUTPUT_PATH / "perplexity_features.csv"
    
    features_df.to_csv(args.output, index=False)
    print(f"\nFeatures saved to: {args.output}")
    print(f"Shape: {features_df.shape}")

if __name__ == "__main__":
    main()