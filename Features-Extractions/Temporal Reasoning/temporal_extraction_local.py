# -*- coding: utf-8 -*-
"""
Temporal Components Extraction Pipeline for AI Text Detection
Requires GPU support + trained temporal relation classifier
"""

import os
import sys
import time
import pickle
import shutil
import subprocess
import argparse
import json
import urllib.request
import zipfile
import socket
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F


# CONFIGURATION


class Config:
    CORENLP_VERSION = "4.5.4"
    CHECKPOINT_INTERVAL = 50
    BATCH_SIZE = 64
    MAX_SEQUENCE_LENGTH = 256
    RELATION_WINDOW = 5  # only classify nearby events


# UTILITY FUNCTIONS


def check_java():
    """Check if Java is installed"""
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True, stderr=subprocess.STDOUT)
        return True
    except:
        return False

def check_gpu():
    """Check GPU availability and print info"""
    print("\n" + "="*60)
    print("SYSTEM CHECK")
    print("="*60)
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return torch.device('cuda')
    else:
        print("WARNING: No GPU detected, using CPU")
        return torch.device('cpu')

def setup_directories(base_path: str) -> Dict[str, str]:
    """Create directory structure for outputs"""
    dirs = {
        'base': base_path,
        'checkpoints': os.path.join(base_path, 'checkpoints'),
        'events': os.path.join(base_path, 'events'),
        'timex': os.path.join(base_path, 'timex'),
        'tam': os.path.join(base_path, 'tense_aspect_mood'),
        'relations': os.path.join(base_path, 'relations'),
        'graphs': os.path.join(base_path, 'graphs'),
        'analysis': os.path.join(base_path, 'analysis')
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def save_checkpoint(data: dict, checkpoint_path: str, progress_path: str, progress_idx: int):
    """Save checkpoint with atomic write"""
    temp_checkpoint = checkpoint_path + '.tmp'
    temp_progress = progress_path + '.tmp'
    with open(temp_checkpoint, 'wb') as f:
        pickle.dump(data, f)
    with open(temp_progress, 'w') as f:
        f.write(str(progress_idx))
    shutil.move(temp_checkpoint, checkpoint_path)
    shutil.move(temp_progress, progress_path)

def load_checkpoint(checkpoint_path: str, progress_path: str) -> Tuple[dict, int]:
    """Load checkpoint if exists"""
    if os.path.exists(checkpoint_path) and os.path.exists(progress_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        with open(progress_path, 'r') as f:
            start_idx = int(f.read().strip())
        return data, start_idx
    return {}, 0


# CORENLP SETUP


def download_corenlp(install_dir: str) -> str:
    """Download Stanford CoreNLP from GitHub"""
    corenlp_dir = os.path.join(install_dir, f"stanford-corenlp-{Config.CORENLP_VERSION}")
    
    if os.path.exists(corenlp_dir):
        print(f"CoreNLP already exists at: {corenlp_dir}")
        return corenlp_dir
    
    print("Downloading CoreNLP from GitHub...")
    github_url = f"https://github.com/stanfordnlp/CoreNLP/releases/download/v{Config.CORENLP_VERSION}/stanford-corenlp-{Config.CORENLP_VERSION}.zip"
    zip_path = os.path.join(install_dir, "stanford-corenlp.zip")
    
    socket.setdefaulttimeout(300)
    urllib.request.urlretrieve(github_url, zip_path)
    
    print("Extracting CoreNLP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(install_dir)
    os.remove(zip_path)
    
    return corenlp_dir

def start_corenlp_server(corenlp_dir: str):
    """Start CoreNLP server for SUTime"""
    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP(corenlp_dir, memory='4g', timeout=30000)
    return nlp


# DATA LOADING


def load_dataset(dataset_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load dataset and create texts dictionary"""
    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    required_cols = ['id', 'is_ai']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    text_col = 'generation' if 'generation' in df.columns else 'text'
    if text_col not in df.columns:
        raise ValueError("No text column found (expected 'generation' or 'text')")
    
    texts_dict = {row['id']: row[text_col] for _, row in df.iterrows()}
    
    print(f"Loaded {len(df)} documents (AI: {(df['is_ai']==1).sum()}, Human: {(df['is_ai']==0).sum()})")
    return df, texts_dict


# EVENT EXTRACTION (STANZA)


def setup_stanza(device: torch.device):
    """Initialize Stanza pipeline"""
    import stanza
    stanza.download('en', verbose=False)
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', 
                          use_gpu=device.type=='cuda', verbose=False)
    return nlp

def extract_events_stanza(nlp_stanza, text: str, doc_id: str) -> List[Dict]:
    """Extract events using Stanza"""
    doc = nlp_stanza(text)
    events = []
    
    for sent_idx, sent in enumerate(doc.sentences):
        for word in sent.words:
            if word.upos == 'VERB':
                feats = {}
                if word.feats:
                    for feat in word.feats.split('|'):
                        if '=' in feat:
                            key, val = feat.split('=')
                            feats[key] = val
                
                arguments = [{'role': w.deprel, 'text': w.text, 'upos': w.upos} 
                            for w in sent.words if w.head == word.id]
                
                position = text.find(word.text)
                events.append({
                    'doc_id': doc_id, 'sentence_id': sent_idx,
                    'trigger': word.text, 'lemma': word.lemma,
                    'trigger_start': position,
                    'trigger_end': position + len(word.text) if position >= 0 else -1,
                    'type': 'Event',
                    'tense': feats.get('Tense', 'None'),
                    'aspect': feats.get('Aspect', 'None'),
                    'mood': feats.get('Mood', 'None'),
                    'voice': feats.get('Voice', 'None'),
                    'arguments': arguments, 'confidence': 1.0
                })
    return events

def run_event_extraction(nlp_stanza, texts_dict: Dict, dirs: Dict) -> Dict:
    """Run event extraction on all documents"""
    checkpoint_path = os.path.join(dirs['checkpoints'], 'stanza_events.pkl')
    progress_path = os.path.join(dirs['checkpoints'], 'stanza_progress.txt')
    
    events_dict, start_idx = load_checkpoint(checkpoint_path, progress_path)
    doc_ids = list(texts_dict.keys())
    
    print(f"\nExtracting events from {len(doc_ids)} documents (starting from {start_idx})...")
    
    for i in tqdm(range(start_idx, len(doc_ids)), desc="Stanza extraction"):
        doc_id = doc_ids[i]
        try:
            events_dict[doc_id] = extract_events_stanza(nlp_stanza, texts_dict[doc_id], doc_id)
        except Exception as e:
            print(f"\nError on {doc_id}: {e}")
            events_dict[doc_id] = []
        
        if (i + 1) % Config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(events_dict, checkpoint_path, progress_path, i + 1)
    
    final_path = os.path.join(dirs['events'], 'stanza_events_final.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(events_dict, f)
    
    return events_dict


# TEMPORAL EXPRESSION EXTRACTION (SUTIME)


def extract_temporal_expressions(nlp_corenlp, text: str, doc_id: str) -> List[Dict]:
    """Extract temporal expressions using SUTime"""
    result = nlp_corenlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,ner', 'outputFormat': 'json'
    })
    if isinstance(result, str):
        result = json.loads(result)
    
    timexes = []
    for sentence in result.get('sentences', []):
        for token in sentence.get('tokens', []):
            if token.get('ner', '') in ['DATE', 'TIME', 'DURATION', 'SET']:
                timexes.append({
                    'doc_id': doc_id, 'text': token['word'], 'type': token['ner'],
                    'position': token.get('characterOffsetBegin', -1),
                    'normalized': token.get('normalizedNER', None)
                })
    return timexes

def run_timex_extraction(nlp_corenlp, texts_dict: Dict, dirs: Dict) -> Dict:
    """Run temporal expression extraction on all documents"""
    checkpoint_path = os.path.join(dirs['checkpoints'], 'sutime_timex.pkl')
    progress_path = os.path.join(dirs['checkpoints'], 'sutime_progress.txt')
    
    timex_dict, start_idx = load_checkpoint(checkpoint_path, progress_path)
    doc_ids = list(texts_dict.keys())
    
    print(f"\nExtracting temporal expressions from {len(doc_ids)} documents...")
    
    for i in tqdm(range(start_idx, len(doc_ids)), desc="SUTime extraction"):
        doc_id = doc_ids[i]
        try:
            timex_dict[doc_id] = extract_temporal_expressions(nlp_corenlp, texts_dict[doc_id], doc_id)
        except Exception as e:
            print(f"\nError on {doc_id}: {e}")
            timex_dict[doc_id] = []
        
        if (i + 1) % Config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(timex_dict, checkpoint_path, progress_path, i + 1)
    
    final_path = os.path.join(dirs['timex'], 'sutime_timex_final.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(timex_dict, f)
    
    return timex_dict


# BINARY TEMPORAL RELATIONS
# REQUESTED ALREADY TRAINED MODEL > CHECK .IPYNB temporal reasoning model


ID2LABEL = {0: 'BEFORE', 1: 'AFTER'}
LABEL2ID = {'BEFORE': 0, 'AFTER': 1}

def load_temporal_classifier(model_dir: str, device: torch.device):
    """Load trained binary temporal classifier"""
    from transformers import RobertaForSequenceClassification, RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer

def insert_markers(context: str, e1_text: str, e2_text: str, p1: int, p2: int) -> str:
    """Insert event markers into text"""
    if p1 >= 0 and p2 >= 0:
        if p1 < p2:
            return (context[:p1] + f"<e1> {e1_text} </e1>" + context[p1+len(e1_text):p2] + 
                   f"<e2> {e2_text} </e2>" + context[p2+len(e2_text):])
        else:
            return (context[:p2] + f"<e2> {e2_text} </e2>" + context[p2+len(e2_text):p1] + 
                   f"<e1> {e1_text} </e1>" + context[p1+len(e1_text):])
    marked = context.replace(e1_text, f"<e1> {e1_text} </e1>", 1)
    return marked.replace(e2_text, f"<e2> {e2_text} </e2>", 1)

def predict_relations_batch(model, tokenizer, relation_inputs: List, device: torch.device) -> List:
    """Predict relations in batches"""
    if not relation_inputs:
        return []
    
    results = []
    use_amp = device.type == 'cuda'
    
    for i in range(0, len(relation_inputs), Config.BATCH_SIZE):
        batch = relation_inputs[i:i+Config.BATCH_SIZE]
        marked_texts = [insert_markers(ctx, e1, e2, p1, p2) for ctx, e1, e2, p1, p2 in batch]
        
        encoded = tokenizer(marked_texts, truncation=True, max_length=Config.MAX_SEQUENCE_LENGTH,
                           padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1)
        
        for pred, prob in zip(preds, probs):
            results.append((ID2LABEL[pred], prob))
    
    return results

def extract_binary_relations(model, tokenizer, text: str, events: List[Dict], 
                            doc_id: str, device: torch.device) -> List[Dict]:
    """Extract binary temporal relations for a document"""
    relation_inputs, event_pairs = [], []
    
    for i, event1 in enumerate(events[:-1]):
        for event2 in events[i+1:min(i+Config.RELATION_WINDOW+1, len(events))]:
            relation_inputs.append((text, event1['trigger'], event2['trigger'],
                                   event1.get('trigger_start', -1), event2.get('trigger_start', -1)))
            event_pairs.append((event1, event2))
    
    predictions = predict_relations_batch(model, tokenizer, relation_inputs, device)
    
    relations = []
    for (event1, event2), (relation, probs) in zip(event_pairs, predictions):
        relations.append({
            'doc_id': doc_id,
            'event1': event1['trigger'], 'event1_lemma': event1['lemma'],
            'event2': event2['trigger'], 'event2_lemma': event2['lemma'],
            'relation': relation,
            'confidence': float(probs[LABEL2ID[relation]]),
            'probs': probs.tolist()
        })
    return relations

def run_relation_extraction(model, tokenizer, texts_dict: Dict, events_dict: Dict, 
                           dirs: Dict, device: torch.device) -> Dict:
    """Run relation extraction on all documents"""
    checkpoint_path = os.path.join(dirs['checkpoints'], 'binary_relations.pkl')
    progress_path = os.path.join(dirs['checkpoints'], 'binary_relations_progress.txt')
    
    relations_dict, start_idx = load_checkpoint(checkpoint_path, progress_path)
    doc_ids = list(events_dict.keys())
    
    print(f"\nExtracting binary temporal relations from {len(doc_ids)} documents...")
    
    for i in tqdm(range(start_idx, len(doc_ids)), desc="Relation extraction"):
        doc_id = doc_ids[i]
        events = events_dict[doc_id]
        
        if len(events) < 2:
            relations_dict[doc_id] = []
            continue
        
        try:
            relations_dict[doc_id] = extract_binary_relations(
                model, tokenizer, texts_dict[doc_id], events, doc_id, device)
        except Exception as e:
            print(f"\nError on {doc_id}: {e}")
            relations_dict[doc_id] = []
        
        if device.type == 'cuda' and (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
        
        if (i + 1) % Config.CHECKPOINT_INTERVAL == 0:
            save_checkpoint(relations_dict, checkpoint_path, progress_path, i + 1)
    
    final_path = os.path.join(dirs['relations'], 'binary_relations_final.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(relations_dict, f)
    
    return relations_dict


# GRAPH CONSTRUCTION


def build_raw_temporal_graph(relations: List[Dict]) -> Optional[nx.DiGraph]:
    """Build directed graph from binary temporal relations (may have cycles)"""
    if not relations:
        return None
    
    G = nx.DiGraph()
    for rel in relations:
        e1, e2 = rel['event1'], rel['event2']
        G.add_node(e1)
        G.add_node(e2)
        if rel['relation'] == 'BEFORE':
            G.add_edge(e1, e2, weight=rel['confidence'], relation='BEFORE')
        elif rel['relation'] == 'AFTER':
            G.add_edge(e2, e1, weight=rel['confidence'], relation='AFTER')
    return G

def build_greedy_temporal_graph(events: List[Dict], relations: List[Dict], doc_id: str) -> Optional[nx.DiGraph]:
    """Build acyclic temporal graph using greedy maximum-weight DAG algorithm"""
    if len(events) < 2 or not relations:
        return None
    
    event_triggers = [e['trigger'] for e in events]
    event_to_idx = {trigger: i for i, trigger in enumerate(event_triggers)}
    
    edges = []
    for rel in relations:
        e1, e2 = rel['event1'], rel['event2']
        if e1 not in event_to_idx or e2 not in event_to_idx:
            continue
        i, j = event_to_idx[e1], event_to_idx[e2]
        if i == j:
            continue
        
        score, conf = rel['confidence'], rel['confidence']
        if rel['relation'] == 'BEFORE':
            edges.append((score, i, j, conf))
        elif rel['relation'] == 'AFTER':
            edges.append((score, j, i, conf))
    
    if not edges:
        return None
    
    edges.sort(key=lambda x: x[0], reverse=True)
    
    G = nx.DiGraph()
    for i, trigger in enumerate(event_triggers):
        G.add_node(trigger, event_idx=i)
    
    n_selected, total_score = 0, 0.0
    for score, i, j, conf in edges:
        u, v = event_triggers[i], event_triggers[j]
        if G.has_edge(u, v):
            continue
        G.add_edge(u, v, relation='BEFORE', confidence=conf, score=score)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(u, v)
        else:
            n_selected += 1
            total_score += score
    
    if n_selected == 0:
        return None
    
    topo_order = {node: rank for rank, node in enumerate(nx.topological_sort(G))}
    for node, rank in topo_order.items():
        G.nodes[node]['topo_order'] = rank
    
    G.graph.update({
        'doc_id': doc_id, 'n_events': len(event_triggers),
        'n_candidate_edges': len(edges), 'n_selected_edges': G.number_of_edges(),
        'edge_retention_rate': G.number_of_edges() / len(edges) if edges else 0,
        'total_score': total_score, 'is_dag': True
    })
    return G

def run_graph_construction(events_dict: Dict, relations_dict: Dict, dirs: Dict) -> Tuple[Dict, Dict]:
    """Build both raw and greedy temporal graphs"""
    print("\nBuilding temporal graphs...")
    
    raw_graphs, greedy_graphs = {}, {}
    
    for doc_id in tqdm(relations_dict.keys(), desc="Building graphs"):
        relations = relations_dict[doc_id]
        events = events_dict[doc_id]
        
        raw_graphs[doc_id] = build_raw_temporal_graph(relations)
        greedy_graphs[doc_id] = build_greedy_temporal_graph(events, relations, doc_id)
    
    with open(os.path.join(dirs['graphs'], 'temporal_graphs_raw_final.pkl'), 'wb') as f:
        pickle.dump(raw_graphs, f)
    with open(os.path.join(dirs['graphs'], 'greedy_temporal_graphs_final.pkl'), 'wb') as f:
        pickle.dump(greedy_graphs, f)
    
    return raw_graphs, greedy_graphs


# VALIDATION


def validate_pipeline(events_dict: Dict, timex_dict: Dict, relations_dict: Dict,
                     raw_graphs: Dict, greedy_graphs: Dict, texts_dict: Dict):
    """Validate extracted components"""
    print("\n" + "="*60)
    print("PIPELINE VALIDATION")
    print("="*60)
    
    total_events = sum(len(e) for e in events_dict.values())
    total_timex = sum(len(t) for t in timex_dict.values())
    total_relations = sum(len(r) for r in relations_dict.values())
    
    print(f"\nEvents: {total_events:,} total, {np.mean([len(e) for e in events_dict.values()]):.1f} per doc")
    print(f"Temporal expressions: {total_timex:,} total")
    print(f"Relations: {total_relations:,} total")
    
    non_empty_raw = [g for g in raw_graphs.values() if g is not None]
    non_empty_greedy = [g for g in greedy_graphs.values() if g is not None]
    
    raw_cyclic = sum(1 for g in non_empty_raw if not nx.is_directed_acyclic_graph(g))
    greedy_cyclic = sum(1 for g in non_empty_greedy if not nx.is_directed_acyclic_graph(g))
    
    print(f"\nRaw graphs: {len(non_empty_raw)} non-empty, {raw_cyclic} with cycles")
    print(f"Greedy graphs: {len(non_empty_greedy)} non-empty, {greedy_cyclic} with cycles")
    
    if greedy_cyclic == 0:
        print("All greedy graphs are DAGs")
    
    sample_doc = next((d for d in events_dict if len(events_dict[d]) > 0 and 
                       len(relations_dict[d]) > 0), None)
    if sample_doc:
        print(f"\nSample document: {sample_doc}")
        print(f"  Text: {texts_dict[sample_doc][:100]}...")
        print(f"  Events: {len(events_dict[sample_doc])}")
        print(f"  Relations: {len(relations_dict[sample_doc])}")
        if events_dict[sample_doc]:
            e = events_dict[sample_doc][0]
            print(f"  First event: {e['trigger']} (tense={e['tense']})")
        if relations_dict[sample_doc]:
            r = relations_dict[sample_doc][0]
            print(f"  First relation: {r['event1']} {r['relation']} {r['event2']} (conf={r['confidence']:.3f})")


# MAIN


def main():
    parser = argparse.ArgumentParser(description='Temporal Components Extraction Pipeline')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--output', type=str, default='./temporal_extraction_output', help='Output directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained temporal classifier')
    parser.add_argument('--corenlp', type=str, default='./corenlp', help='CoreNLP installation directory')
    parser.add_argument('--skip-timex', action='store_true', help='Skip SUTime extraction')
    args = parser.parse_args()
    
    device = check_gpu()
    
    if not check_java():
        print("ERROR: Java not found. Please install Java for CoreNLP.")
        if not args.skip_timex:
            sys.exit(1)
    
    dirs = setup_directories(args.output)
    print(f"Output directory: {args.output}")
    
    df, texts_dict = load_dataset(args.dataset)
    
    print("\nInitializing Stanza...")
    nlp_stanza = setup_stanza(device)
    
    print("\nLoading temporal classifier...")
    model, tokenizer = load_temporal_classifier(args.model, device)
    
    events_dict = run_event_extraction(nlp_stanza, texts_dict, dirs)
    
    if not args.skip_timex:
        print("\nStarting CoreNLP server...")
        corenlp_dir = download_corenlp(args.corenlp)
        nlp_corenlp = start_corenlp_server(corenlp_dir)
        timex_dict = run_timex_extraction(nlp_corenlp, texts_dict, dirs)
        nlp_corenlp.close()
    else:
        timex_dict = {doc_id: [] for doc_id in texts_dict}
        print("\nSkipping SUTime extraction")
    
    relations_dict = run_relation_extraction(model, tokenizer, texts_dict, events_dict, dirs, device)
    
    raw_graphs, greedy_graphs = run_graph_construction(events_dict, relations_dict, dirs)
    
    validate_pipeline(events_dict, timex_dict, relations_dict, raw_graphs, greedy_graphs, texts_dict)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {args.output}")

if __name__ == '__main__':
    main()