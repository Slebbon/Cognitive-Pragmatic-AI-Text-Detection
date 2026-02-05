# Cognitively-Pragmatic Grounded AI Text Detection

This repository accompanies a master’s thesis on **AI-generated text detection**, proposing a cognitively and pragmatically grounded approach to intrinsic detection.  
The work operationalises linguistic, cognitive, and discourse-level theories into computational features.

The project focuses on **interpretability** and **feature-level analysis** rather than end-to-end black-box detection alone.

---

## Research Objectives

The main goals of this project are:

- To design a feature extraction pipeline grounded in linguistic, cognitive, and pragmatic theory
- To evaluate the eventual discriminative power of these features for AI-text detection
- To analyse feature complementarity via ablation and incremental integration
- To preserve interpretability by using decomposable, feature-based models


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


------ 

## Model Dependencies

Some feature families depend on pretrained models, which are provided for convenience:

#### Pairwise Temporal Classifier

Temporal reasoning features rely on a pretrained pairwise temporal model.  
A compatible binary checkpoint is included under: models/roberta-matres-binary/

#### Detection Models

Pretrained versions of the main detection models proposed in the thesis'conclusion are also included:

- Mono Cognitive–Pragmatic Model  
- Full Integrated Model  

These models are provided to support qualitative inspection and approximate replication of the reported experiments. Minor numerical discrepancies with the thesis results may arise due to preprocessing choices, random seeds, or platform-dependent factors.

---

## Reproducibility Notes and Limitations

This repository is primarily intended as the **research artefact** accompanying the master’s thesis, with the goal of enabling inspection and reproduction of the proposed methodology.

The main reproducible contribution of this work concerns the **feature extraction pipeline** proposed in the thesis.  
Feature computation can be reproduced starting from raw textual input by using:

- `prototype-ready-to-use.ipynb`, in conjunction with the `feature_extraction/` directory.

The notebook is designed as a reference implementation to demonstrate how the different feature families are computed and combined, and to allow independent inspection or re-use of individual components.

The stylometric feature extraction pipeline is included in full for transparency and completeness. However, it should be considered **exploratory** in nature and treated as an artificially constructed benchmark for the specific scope described in the thesis. Specifically:

- It reflects a baseline configuration corresponding to an early, exploratory consolidation of stylometric features, as discussed in the thesis.
- It is known to be high-variance and sensitive to preprocessing choices and parameter settings.
- It does not fully align with the refined and theoretically consolidated feature inventory adopted in later experimental stages of the thesis.
- Its exact reproducibility may be affected by subsequent feature drift, database updates, and changes in data stratification over the course of the research.

For these reasons, the stylometric pipeline should not be interpreted as a definitive or optimised implementation, nor as an exhaustive benchmark. Rather, it is provided as a **contextual and interpretive reference** to facilitate linguistic and computational analysis of the cognitive-pragmatic pipeline.

When reuse is required, it is recommended to restrict feature extraction to the **core stylometric backbone (7 features)** as a stable baseline, extending it with features from the **optimised baseline features (13 features)** described in the thesis based on the controlled linguistic experimentation wanted.



