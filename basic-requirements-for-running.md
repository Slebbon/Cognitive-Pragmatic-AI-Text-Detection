Some feature families rely on external pretrained models or third-party tools. These dependencies are documented here for completeness.

## Temporal Reasoning Features

Temporal reasoning features require a pairwise binary classifier trained on temporal relation data.  
A reference training and usage notebook is provided in:

`Features-Extraction\Temporal Reasoning\External Models Training\TemporalReasoning_model.ipynb`


For feature extraction, an existing pretrained pairwise temporal classifier can be directly imported and used. The specific model checkpoint employed in the experiments is compatible with the TempEval-style temporal relation schema; alternative compatible models may be substituted without affecting the feature extraction logic available at: `models\temporal-pairwise-model\roberta-matres-binary`.

## Perplexity-Based Features

Perplexity computations rely on the Hugging Face `transformers` library and a pretrained language model.  
In the reported experiments, the Pythia-160M model was used. Equivalent causal language models may be substituted, provided that perplexity is computed consistently.

## Additional Dependencies

Temporal tagging components based on SUTime require a local Java installation.

A Hugging Face access token may optionally be provided to accelerate model downloads, but is not strictly required for execution.

All feature extraction and inspection steps are performed through the provided notebooks.  
Library dependencies are specified in `requirements.txt`.
