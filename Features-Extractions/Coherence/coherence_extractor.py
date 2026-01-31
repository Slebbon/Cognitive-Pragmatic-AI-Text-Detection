# coherence_extractor.py
# entity, semantic, and topic coherence

import pandas as pd
import numpy as np
import pickle
import os
import time
import networkx as nx
import spacy
import torch
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
from scipy.stats import linregress, entropy
from typing import Dict, List, Tuple
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#NOTA BENE: Sul server è possibile questo errore: BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
#Key                     | Status     |  |
#embeddings.position_ids | UNEXPECTED |  |

#può essere ignorato, è un warning di caricamento che non influenza il funzionamento del modello ed è dovuto all'hub di HF usa il token HF_TOKEN del server.

# configuration, can be changed


BASE_PATH = Path(__file__).parent
CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_PATH, "output")

# create directories
for d in [BASE_PATH, CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# bertopic config (NEEDS be adjusted based on dataset size given)
MIN_TOPIC_SIZE = 30
N_NEIGHBORS = 15
BATCH_SIZE = 16
MIN_SENTENCES_FOR_TOPICS = 50  # minimum sentences needed for topic modeling


# gpu setup


def setup_device():
    if torch.cuda.is_available():
        print(f"[GPU] using: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("[CPU] no gpu available, using cpu")
        return "cpu"

DEVICE = setup_device()


# model loading


def load_spacy_model():
    print("loading spacy model...")
    nlp = spacy.load("en_core_web_trf")
    if DEVICE == "cuda":
        spacy.require_gpu()
    nlp.disable_pipes([p for p in nlp.pipe_names if p not in ['transformer', 'tagger', 'parser', 'ner', 'attribute_ruler']])
    print(f"spacy loaded, active pipes: {nlp.pipe_names}")
    return nlp

def load_sentence_transformer():
    print("loading sentence-bert model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    if DEVICE == "cuda":
        model = model.to('cuda')
    print(f"sentence-bert loaded, dim: {model.get_sentence_embedding_dimension()}")
    return model


# entity coherence extraction


def extract_entities_from_doc(doc, doc_id: str) -> Dict:
    # extract entities using ner and noun chunks
    result = {
        'sentences': [],
        'entities_per_sentence': [],
        'all_entities': set(),
        'entity_frequencies': Counter()
    }
    
    sentences = list(doc.sents)
    if len(sentences) == 0:
        return result
    
    token_to_entity = {}
    
    # named entities first
    for ent in doc.ents:
        canonical = ent.text.lower().strip()
        for token in ent:
            token_to_entity[token.i] = canonical
    
    # noun chunks second
    for chunk in doc.noun_chunks:
        if len(chunk) <= 4:
            canonical = chunk.text.lower().strip()
            chunk_tokens = [t.i for t in chunk]
            if not any(t in token_to_entity for t in chunk_tokens):
                for token in chunk:
                    token_to_entity[token.i] = canonical
    
    for sent in sentences:
        sent_entities = set()
        for token in sent:
            if token.i in token_to_entity:
                entity = token_to_entity[token.i]
                sent_entities.add(entity)
                result['entity_frequencies'][entity] += 1
        result['sentences'].append(sent.text.strip())
        result['entities_per_sentence'].append(sent_entities)
        result['all_entities'].update(sent_entities)
    
    return result

def compute_entity_features(doc_id: str, entity_info: Dict) -> Dict:
    # compute 6 entity cohesion features
    features = {'id': doc_id}
    
    sentences = entity_info['sentences']
    entities_per_sentence = entity_info['entities_per_sentence']
    all_entities = entity_info['all_entities']
    entity_frequencies = entity_info['entity_frequencies']
    n_sentences = len(sentences)
    
    if n_sentences < 2 or len(all_entities) == 0:
        return create_empty_entity_features(doc_id)
    
    total_tokens = sum(len(sent.split()) for sent in sentences)
    total_mentions = sum(entity_frequencies.values())
    n_unique_entities = len(all_entities)
    
    # feature 1: entity mention density
    features['entity_mention_density'] = (total_mentions / total_tokens) * 1000 if total_tokens > 0 else 0
    
    # feature 2: entity reuse rate
    features['entity_reuse_rate'] = (total_mentions - n_unique_entities) / total_mentions if total_mentions > 0 else 0
    
    # feature 3: entity graph density
    G = nx.Graph()
    G.add_nodes_from(range(n_sentences))
    for i in range(n_sentences):
        for j in range(i+1, n_sentences):
            shared = entities_per_sentence[i] & entities_per_sentence[j]
            if len(shared) > 0:
                G.add_edge(i, j)
    max_edges = (n_sentences * (n_sentences - 1)) / 2
    features['entity_graph_density'] = G.number_of_edges() / max_edges if max_edges > 0 else 0
    
    # feature 4: isolated sentences
    degrees = dict(G.degree())
    isolated = sum(1 for deg in degrees.values() if deg == 0)
    features['entity_isolated_sentences'] = isolated / n_sentences
    
    # feature 5: largest connected component
    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        if components:
            largest = max(components, key=len)
            features['entity_largest_component_size'] = len(largest) / n_sentences
        else:
            features['entity_largest_component_size'] = 0.0
    else:
        features['entity_largest_component_size'] = 0.0
    
    # feature 6: mean entity continuation rate
    continuation_rates = []
    for i in range(n_sentences - 1):
        if len(entities_per_sentence[i]) > 0:
            continuing = len(entities_per_sentence[i] & entities_per_sentence[i+1])
            rate = continuing / len(entities_per_sentence[i])
            continuation_rates.append(rate)
    features['mean_entity_continuation_rate'] = np.mean(continuation_rates) if continuation_rates else 0.0
    
    return features

def create_empty_entity_features(doc_id: str) -> Dict:
    return {
        'id': doc_id,
        'entity_mention_density': np.nan,
        'entity_reuse_rate': np.nan,
        'entity_graph_density': np.nan,
        'entity_isolated_sentences': np.nan,
        'entity_largest_component_size': np.nan,
        'mean_entity_continuation_rate': np.nan
    }


# semantic coherence extraction


def compute_semantic_features(doc_id: str, embeddings: np.ndarray) -> Dict:
    # compute 10 semantic cohesion features
    features = {'id': doc_id}
    n_sentences = len(embeddings)
    
    if n_sentences < 2:
        return create_empty_semantic_features(doc_id)
    
    # compute similarity matrix
    sim_matrix = np.zeros((n_sentences, n_sentences))
    for i in range(n_sentences):
        for j in range(i+1, n_sentences):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    
    # feature 1: mean adjacent cosine similarity
    adjacent_sims = [sim_matrix[i, i+1] for i in range(n_sentences-1)]
    features['mean_adjacent_cosine_similarity'] = np.mean(adjacent_sims)
    
    # feature 2: min adjacent cosine similarity
    features['min_adjacent_cosine_similarity'] = np.min(adjacent_sims)
    
    # feature 3: adjacent similarity variance
    features['adjacent_similarity_variance'] = np.var(adjacent_sims)
    
    # feature 4: semantic graph density
    threshold = 0.6
    adj_matrix = (sim_matrix >= threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)
    n_edges = np.sum(adj_matrix) / 2
    max_edges = (n_sentences * (n_sentences - 1)) / 2
    features['semantic_graph_density'] = n_edges / max_edges if max_edges > 0 else 0
    
    # feature 5: similarity decay rate
    sims_by_distance = {}
    for dist in range(1, n_sentences):
        sims = [sim_matrix[i, i+dist] for i in range(n_sentences-dist)]
        if sims:
            sims_by_distance[dist] = np.mean(sims)
    if len(sims_by_distance) >= 2:
        dists = list(sims_by_distance.keys())
        means = list(sims_by_distance.values())
        slope, _, _, _, _ = linregress(dists, means)
        features['similarity_decay_rate'] = slope
    else:
        features['similarity_decay_rate'] = 0.0
    
    # feature 6: mean non-adjacent similarity
    nonadj_sims = []
    for dist in [2, 3]:
        if dist < n_sentences:
            sims = [sim_matrix[i, i+dist] for i in range(n_sentences-dist)]
            nonadj_sims.extend(sims)
    features['mean_nonadjacent_similarity'] = np.mean(nonadj_sims) if nonadj_sims else np.nan
    
    # feature 7: long-range similarity
    longrange_sims = []
    for i in range(n_sentences):
        for j in range(i+4, n_sentences):
            longrange_sims.append(sim_matrix[i, j])
    features['long_range_similarity'] = np.mean(longrange_sims) if longrange_sims else np.nan
    
    # build graph for topology features
    G = nx.Graph()
    G.add_nodes_from(range(n_sentences))
    for i in range(n_sentences):
        for j in range(i+1, n_sentences):
            if sim_matrix[i, j] >= threshold:
                G.add_edge(i, j)
    
    # feature 8: isolated sentences
    degrees = dict(G.degree())
    isolated = sum(1 for deg in degrees.values() if deg == 0)
    features['semantic_graph_isolated_sentences'] = isolated / n_sentences
    
    # feature 9: largest connected component
    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        if components:
            largest = max(components, key=len)
            features['semantic_largest_component_size'] = len(largest) / n_sentences
        else:
            features['semantic_largest_component_size'] = 0.0
    else:
        features['semantic_largest_component_size'] = 0.0
    
    # feature 10: average degree
    features['semantic_average_degree'] = np.mean(list(degrees.values()))
    
    return features

def create_empty_semantic_features(doc_id: str) -> Dict:
    return {
        'id': doc_id,
        'mean_adjacent_cosine_similarity': np.nan,
        'min_adjacent_cosine_similarity': np.nan,
        'adjacent_similarity_variance': np.nan,
        'semantic_graph_density': np.nan,
        'similarity_decay_rate': np.nan,
        'mean_nonadjacent_similarity': np.nan,
        'long_range_similarity': np.nan,
        'semantic_graph_isolated_sentences': np.nan,
        'semantic_largest_component_size': np.nan,
        'semantic_average_degree': np.nan
    }


# topic coherence extraction


def train_bertopic(all_sentences: List[str], all_embeddings: np.ndarray) -> Tuple[BERTopic, List[int]]:
    # train bertopic with adaptive configuration based on dataset size
    print("training bertopic model...")
    
    n_sentences = len(all_sentences)
    
    # check if we have enough data for topic modeling
    if n_sentences < MIN_SENTENCES_FOR_TOPICS:
        print(f"warning: only {n_sentences} sentences, skipping topic modeling (need {MIN_SENTENCES_FOR_TOPICS}+)")
        return None, [-1] * n_sentences
    
    # adaptive parameters based on dataset size
    adaptive_min_topic = max(5, min(MIN_TOPIC_SIZE, n_sentences // 10))
    adaptive_neighbors = max(5, min(N_NEIGHBORS, n_sentences // 5))
    adaptive_min_df = 1 if n_sentences < 100 else 2
    
    print(f"  adaptive config: min_topic={adaptive_min_topic}, neighbors={adaptive_neighbors}, min_df={adaptive_min_df}")
    
    try:
        umap_model = UMAP(
            n_neighbors=adaptive_neighbors, 
            n_components=min(5, n_sentences - 2),  # n_components must be < n_samples
            min_dist=0.0, 
            metric='cosine', 
            random_state=42
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=adaptive_min_topic, 
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True
        )
        
        vectorizer_model = CountVectorizer(
            stop_words='english', 
            min_df=adaptive_min_df,
            max_df=0.95,  # ignore terms in >95% of docs
            ngram_range=(1, 2)
        )
        
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics='auto',
            min_topic_size=adaptive_min_topic,
            top_n_words=10,
            language='english',
            calculate_probabilities=False,
            verbose=False
        )
        
        topics, _ = topic_model.fit_transform(all_sentences, all_embeddings)
        n_topics = len(set(topics) - {-1})
        n_outliers = sum(1 for t in topics if t == -1)
        print(f"bertopic trained: {n_topics} topics, {n_outliers} outliers ({100*n_outliers/len(topics):.1f}%)")
        
        return topic_model, topics
        
    except Exception as e:
        print(f"warning: bertopic failed ({e}), using fallback")
        return None, [-1] * n_sentences

def compute_topic_features(doc_id: str, topics: List[int], embeddings: np.ndarray) -> Dict:
    # compute 10 topic coherence features
    features = {'id': doc_id}
    
    if len(topics) < 2:
        return create_empty_topic_features(doc_id)
    
    topics_filtered = [t for t in topics if t != -1]
    
    # feature 1: topic entropy
    if len(topics_filtered) > 0:
        topic_counts = Counter(topics_filtered)
        total = sum(topic_counts.values())
        probs = [count / total for count in topic_counts.values()]
        features['topic_entropy'] = entropy(probs)
    else:
        features['topic_entropy'] = np.nan
    
    # feature 2: topic drift rate
    transitions = sum(1 for i in range(len(topics)-1) if topics[i] != topics[i+1])
    features['topic_drift_rate'] = transitions / (len(topics) - 1)
    
    # feature 3: dominant topic proportion
    if len(topics_filtered) > 0:
        topic_counts = Counter(topics_filtered)
        most_common = topic_counts.most_common(1)[0][1]
        features['dominant_topic_proportion'] = most_common / len(topics_filtered)
    else:
        features['dominant_topic_proportion'] = np.nan
    
    # feature 4: topic switching frequency
    features['topic_switching_frequency'] = transitions
    
    # feature 5: topic persistence
    spans = []
    current_span = 1
    for i in range(1, len(topics)):
        if topics[i] == topics[i-1]:
            current_span += 1
        else:
            spans.append(current_span)
            current_span = 1
    spans.append(current_span)
    features['topic_persistence'] = np.mean(spans)
    
    # feature 6: num distinct topics
    features['num_distinct_topics'] = len(set(topics_filtered))
    
    # feature 7: topic diversity
    if len(topics_filtered) > 0 and features['topic_entropy'] is not np.nan:
        n_unique = len(set(topics_filtered))
        max_entropy = np.log(n_unique) if n_unique > 1 else 1.0
        features['topic_diversity'] = features['topic_entropy'] / max_entropy if max_entropy > 0 else 0.0
    else:
        features['topic_diversity'] = np.nan
    
    # feature 8: topic concentration (gini)
    if len(topics_filtered) > 0:
        topic_counts = Counter(topics_filtered)
        counts = sorted(topic_counts.values())
        n = len(counts)
        if n > 0:
            cumsum = np.cumsum(counts)
            gini = (2 * np.sum((np.arange(1, n+1) * counts)) / (n * cumsum[-1])) - (n + 1) / n
            features['topic_concentration'] = max(0, min(1, gini))
        else:
            features['topic_concentration'] = np.nan
    else:
        features['topic_concentration'] = np.nan
    
    # feature 9: topic transition similarity
    transition_sims = []
    for i in range(len(topics)-1):
        if topics[i] != topics[i+1] and len(embeddings) > i+1:
            sim = 1 - cosine(embeddings[i], embeddings[i+1])
            transition_sims.append(sim)
    features['topic_transition_similarity'] = np.mean(transition_sims) if transition_sims else np.nan
    
    # feature 10: topic return rate
    if len(topics) >= 3:
        returns = 0
        seen = {topics[0]}
        for i in range(1, len(topics)):
            if topics[i] != topics[i-1] and topics[i] in seen:
                returns += 1
            seen.add(topics[i])
        features['topic_return_rate'] = returns / (len(topics) - 1)
    else:
        features['topic_return_rate'] = 0.0
    
    return features

def create_empty_topic_features(doc_id: str) -> Dict:
    return {
        'id': doc_id,
        'topic_entropy': np.nan,
        'topic_drift_rate': np.nan,
        'dominant_topic_proportion': np.nan,
        'topic_switching_frequency': 0,
        'topic_persistence': np.nan,
        'num_distinct_topics': 0,
        'topic_diversity': np.nan,
        'topic_concentration': np.nan,
        'topic_transition_similarity': np.nan,
        'topic_return_rate': 0.0
    }


# main extraction pipeline


def extract_all_coherence_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    # main pipeline: extract all 26 coherence features
    print("\n" + "="*70)
    print("COHERENCE FEATURE EXTRACTION PIPELINE")
    print("="*70)
    
    # load models
    nlp = load_spacy_model()
    sbert = load_sentence_transformer()
    
    texts_dict = {row['id']: row[text_col] for _, row in df.iterrows()}
    doc_ids = list(texts_dict.keys())
    
    # step 1: process with spacy (entities + sentences)
    print("\n[1/4] processing with spacy...")
    entity_data = {}
    sentence_data = {}
    
    for batch_start in tqdm(range(0, len(doc_ids), BATCH_SIZE), desc="spacy processing"):
        batch_end = min(batch_start + BATCH_SIZE, len(doc_ids))
        batch_ids = doc_ids[batch_start:batch_end]
        batch_texts = [texts_dict[did] for did in batch_ids]
        
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        docs = list(nlp.pipe(batch_texts, batch_size=BATCH_SIZE))
        
        for doc_id, doc in zip(batch_ids, docs):
            entity_data[doc_id] = extract_entities_from_doc(doc, doc_id)
            sentence_data[doc_id] = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # step 2: generate sentence embeddings
    print("\n[2/4] generating sentence embeddings...")
    embeddings_data = {}
    
    for doc_id in tqdm(doc_ids, desc="embedding"):
        sentences = sentence_data[doc_id]
        if len(sentences) > 0:
            embs = sbert.encode(sentences, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
            embeddings_data[doc_id] = embs
        else:
            embeddings_data[doc_id] = np.array([])
    
    # step 3: train bertopic and get topic assignments
    print("\n[3/4] training bertopic...")
    all_sentences = []
    all_embeddings = []
    doc_ids_flat = []
    
    for doc_id in doc_ids:
        if len(embeddings_data[doc_id]) > 0:
            for i, (sent, emb) in enumerate(zip(sentence_data[doc_id], embeddings_data[doc_id])):
                all_sentences.append(sent)
                all_embeddings.append(emb)
                doc_ids_flat.append(doc_id)
    
    all_embeddings = np.array(all_embeddings)
    topic_model, topics = train_bertopic(all_sentences, all_embeddings)
    
    # organize topic assignments by document
    topic_assignments = {}
    for doc_id, topic in zip(doc_ids_flat, topics):
        if doc_id not in topic_assignments:
            topic_assignments[doc_id] = []
        topic_assignments[doc_id].append(topic)
    
    # step 4: extract all features
    print("\n[4/4] extracting features...")
    all_features = []
    
    for doc_id in tqdm(doc_ids, desc="feature extraction"):
        # entity features
        entity_feats = compute_entity_features(doc_id, entity_data[doc_id])
        
        # semantic features
        if len(embeddings_data[doc_id]) >= 2:
            semantic_feats = compute_semantic_features(doc_id, embeddings_data[doc_id])
        else:
            semantic_feats = create_empty_semantic_features(doc_id)
        
        # topic features
        if doc_id in topic_assignments and len(topic_assignments[doc_id]) >= 2:
            topic_feats = compute_topic_features(doc_id, topic_assignments[doc_id], embeddings_data[doc_id])
        else:
            topic_feats = create_empty_topic_features(doc_id)
        
        # merge all features
        combined = {'id': doc_id}
        for feat_dict in [entity_feats, semantic_feats, topic_feats]:
            for k, v in feat_dict.items():
                if k != 'id':
                    combined[k] = v
        
        all_features.append(combined)
    
    features_df = pd.DataFrame(all_features)
    
    # summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"documents processed: {len(features_df)}")
    print(f"features extracted: {len(features_df.columns) - 1}")
    print(f"  - entity coherence: 6 features")
    print(f"  - semantic coherence: 10 features")
    print(f"  - topic coherence: 10 features")
    
    return features_df


# main entry point


def main():
    print("\n" + "="*70)
    print("COHERENCE FEATURE EXTRACTOR")
    print("="*70)
    
    # get input file
    dataset_path = input("\nenter dataset path (csv with 'id' and text column): ").strip()
    
    if not os.path.exists(dataset_path):
        print(f"error: file not found at {dataset_path}")
        return
    
    # load dataset
    print(f"\nloading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"loaded {len(df)} documents")
    
    # check columns
    if 'id' not in df.columns:
        print("error: 'id' column required")
        return
    
    # identify text column
    if 'generation' in df.columns:
        text_col = 'generation'
    elif 'text' in df.columns:
        text_col = 'text'
    else:
        text_col = input("enter text column name: ").strip()
        if text_col not in df.columns:
            print(f"error: column '{text_col}' not found")
            return
    
    print(f"using text column: '{text_col}'")
    
    # extract features
    start_time = time.time()
    features_df = extract_all_coherence_features(df, text_col)
    elapsed = time.time() - start_time
    
    # save output
    output_path = os.path.join(OUTPUT_DIR, "coherence_features.csv")
    features_df.to_csv(output_path, index=False)
    print(f"\nfeatures saved to: {output_path}")
    print(f"total time: {elapsed/60:.1f} minutes")
    
    # display sample
    print("\nsample output (first 3 rows):")
    print(features_df.head(3).to_string())
    
    return features_df


# test with mock data


def create_mock_data(n_samples: int = 20) -> pd.DataFrame:
    # create mock dataset for testing
    np.random.seed(42)
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. The fox was very agile and fast. Dogs are known to be loyal companions.",
        "Machine learning is a subset of artificial intelligence. AI systems can learn from data. Deep learning uses neural networks.",
        "Climate change affects global temperatures. Rising sea levels threaten coastal cities. Environmental policies are crucial.",
        "The stock market fluctuates daily. Investors analyze trends carefully. Economic indicators guide investment decisions.",
        "Python is a popular programming language. Many developers prefer Python for data science. Libraries like pandas are useful.",
    ]
    
    data = []
    for i in range(n_samples):
        text = sample_texts[i % len(sample_texts)]
        # add some variation
        if i % 3 == 0:
            text += " This additional sentence adds more content to the document."
        if i % 4 == 0:
            text += " Another perspective is worth considering here."
        
        data.append({
            'id': f'doc_{i:04d}',
            'text': text,
            'is_ai': i % 2
        })
    
    return pd.DataFrame(data)

def run_test():
    print("\n" + "="*70)
    print("RUNNING TEST WITH MOCK DATA")
    print("="*70)
    
    # create mock data
    mock_df = create_mock_data(20)
    mock_path = os.path.join(BASE_PATH, "mock_data.csv")
    mock_df.to_csv(mock_path, index=False)
    print(f"created mock data: {len(mock_df)} samples")
    print(f"saved to: {mock_path}")
    
    # extract features
    print("\nextracting features from mock data...")
    start_time = time.time()
    features_df = extract_all_coherence_features(mock_df, 'text')
    elapsed = time.time() - start_time
    
    # save test output
    test_output = os.path.join(OUTPUT_DIR, "test_coherence_features.csv")
    features_df.to_csv(test_output, index=False)
    
    # validation
    print("\n" + "="*70)
    print("TEST VALIDATION")
    print("="*70)
    
    # check all columns present
    expected_cols = 27  # id + 6 entity + 10 semantic + 10 topic
    actual_cols = len(features_df.columns)
    print(f"columns: {actual_cols} (expected {expected_cols}) {'PASS' if actual_cols == expected_cols else 'FAIL'}")
    
    # check no all-nan columns
    nan_cols = features_df.columns[features_df.isna().all()].tolist()
    print(f"all-nan columns: {len(nan_cols)} {'PASS' if len(nan_cols) == 0 else 'WARN: ' + str(nan_cols)}")
    
    # check value ranges
    print("\nfeature ranges:")
    for col in features_df.columns:
        if col != 'id':
            min_v = features_df[col].min()
            max_v = features_df[col].max()
            mean_v = features_df[col].mean()
            print(f"  {col}: [{min_v:.4f}, {max_v:.4f}], mean={mean_v:.4f}")
    
    print(f"\ntest completed in {elapsed:.1f} seconds")
    print(f"output saved to: {test_output}")
    
    return features_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test()
    else:
        main()