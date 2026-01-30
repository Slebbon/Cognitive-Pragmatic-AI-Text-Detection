# Cognitively-Grounded AI Text Detection

This repository accompanies a master’s thesis on **AI-generated text detection**, proposing a cognitively and pragmatically grounded approach to intrinsic detection.  
The work operationalises linguistic, cognitive, and discourse-level theories into computational features and evaluates their contribution beyond traditional stylometric baselines.

The project focuses on **interpretability** and **feature-level analysis** rather than end-to-end black-box detection alone.

---

## Research Objectives

The main goals of this project are:

- To design a feature extraction pipeline grounded in linguistic, cognitive, and pragmatic theory
- To evaluate the discriminative power of these features for AI-text detection
- To analyse feature complementarity via ablation and incremental integration
- To preserve interpretability by using decomposable, feature-based models
-

---

## Feature Families

The implemented features are grouped into the following families:

- **Stylometric Baseline**
  - Lexical diversity
  - Character- and word-level statistics
  - Syntactic complexity measures

- **Coreference and Referring Expression**
  - Referring expression generation and coreference

- **Perplexity-based**
  - Perturbation perplexity
  - Global perplexity

- **Metacognition and Metadiscourse**
  - Metacognitive markers
  - Calibration 

- **Discourse and Coherence Features**
  - Semantic cohesion
  - Entity coherence
  - Topic variation

- **Temporal Reasoning Features**
  - Event ordering
  - Temporal graph structure
  - Temporal entropy and constraint violations
