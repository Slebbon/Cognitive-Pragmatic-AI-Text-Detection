# optimization.py
# RandomizedSearchCV for XGBoost on stylometric v1.1 features (27 features)
# comprehensive parameter space covering all XGBoost hyperparameters, n_iter=150


import sys
import argparse
import warnings
import time
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import uniform, randint, loguniform

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import xgboost as xgb

# path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
UTILS_DIR = PROJECT_ROOT / "Utils"
sys.path.insert(0, str(UTILS_DIR))

from feature_utils import impute_missing, cap_extreme_features, verify_features


# feature list (mirrors Test_v11_stylometric_extractor.py SELECTED_FEATURES)


SELECTED_FEATURES = [
    # lexical
    "type_token_ratio",
    "yules_k",
    "hapax_legomena_ratio",
    "token_burstiness",
    "trigram_diversity",
    "hapax_type_ratio",
    # character
    "char_trigram_entropy",
    "compression_ratio",
    "avg_word_length",
    # functional
    "stopword_ratio",
    "comma_ratio",
    # structural
    "sentence_length_std",
    "avg_sentence_length",
    # readability
    "flesch_reading_ease",
    # syntactic
    "avg_tree_depth",
    "avg_dependency_distance",
    "left_dependency_ratio",
    # pos
    "content_function_ratio",
    "verbs_per_100",
    "pos_ratio_CCONJ",
    "pos_transition_entropy",
    "prop_sents_with_verb",
    "noun_verb_ratio",
    "upos_entropy",
    "mean_verbs_per_sent",
    # sentiment
    "sentiment_subjectivity",
    "sentiment_polarity_variance",
]

METADATA_COLS = ["id", "n_tokens_doc", "n_sentences_doc"]
EXCLUDE_COLS = METADATA_COLS + ["is_ai"]

DEFAULT_FEATURES_CSV = (
    PROJECT_ROOT
    / "Features-Extractions"
    / "Stylometric"
    / "output"
    / "stylometric_features_27.csv"
)

N_ITER = 150
CV_FOLDS = 5


# XGBoost fixed params (not searched)


XGB_FIXED = {
    "n_jobs": -1,
    "eval_metric": "logloss",
    "verbosity": 0,
}


# XGBoost parameter space (all tunable hyperparameters)


XGB_PARAM_SPACE = {
    # --- learning ---
    # number of boosting rounds
    "n_estimators": randint(100, 2001),
    # step size shrinkage; low values require more trees
    "learning_rate": loguniform(0.005, 0.5),

    # --- tree structure ---
    # maximum depth of each tree; controls model complexity
    "max_depth": randint(3, 13),
    # minimum sum of instance weight in a child; higher = more conservative
    "min_child_weight": randint(1, 11),
    # minimum loss reduction required to make a further split (pruning)
    "gamma": uniform(0.0, 5.0),
    # maximum delta step for weight update; helps with imbalanced classes
    "max_delta_step": randint(0, 11),

    # --- sampling (stochastic gradient boosting) ---
    # fraction of training samples used per tree
    "subsample": uniform(0.5, 0.5),          # [0.5, 1.0]
    # fraction of features used per tree
    "colsample_bytree": uniform(0.5, 0.5),   # [0.5, 1.0]
    # fraction of features used per level
    "colsample_bylevel": uniform(0.5, 0.5),  # [0.5, 1.0]
    # fraction of features used per split
    "colsample_bynode": uniform(0.5, 0.5),   # [0.5, 1.0]

    # --- regularization ---
    # L1 regularization on leaf weights (sparsity)
    "reg_alpha": loguniform(1e-5, 10.0),
    # L2 regularization on leaf weights (smoothing)
    "reg_lambda": loguniform(1e-3, 10.0),

    # --- tree construction algorithm ---
    # 'hist' is fast and memory-efficient; 'exact' is more precise on small data
    "tree_method": ["hist", "exact"],
}


# data loading and preparation


def load_and_prepare(
    features_csv: str,
    labels_csv: str = None,
    label_col: str = "is_ai",
    id_col: str = "id",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load stylometric features, optionally merge external labels, and clean.

    args:
        features_csv: path to stylometric_features_27.csv
        labels_csv:   optional CSV with id + is_ai (use when features CSV has no labels)
        label_col:    label column name (default 'is_ai')
        id_col:       document id column (default 'id')

    returns:
        cleaned DataFrame with features + label
    """
    features_path = Path(features_csv)
    if not features_path.exists():
        raise FileNotFoundError(f"features CSV not found: {features_path}")

    print(f"\nloading features from: {features_path}")
    df = pd.read_csv(features_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # merge external labels if needed
    if labels_csv is not None:
        labels_path = Path(labels_csv)
        if not labels_path.exists():
            raise FileNotFoundError(f"labels CSV not found: {labels_path}")
        print(f"  merging labels from: {labels_path}")
        labels_df = pd.read_csv(labels_path)[[id_col, label_col]].drop_duplicates()
        before = len(df)
        df = df.merge(labels_df, on=id_col, how="inner")
        print(f"  {len(df)} rows after merge (dropped {before - len(df)} unmatched)")
    elif label_col not in df.columns:
        raise ValueError(
            f"label column '{label_col}' not found in features CSV. "
            f"pass --labels-csv pointing to a file with '{id_col}' and '{label_col}'."
        )

    # restrict to known features + metadata + label
    available = [f for f in SELECTED_FEATURES if f in df.columns]
    missing_feats = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing_feats:
        print(f"  WARNING: {len(missing_feats)} features absent from CSV: {missing_feats}")

    keep_cols = [
        c for c in [id_col, label_col] + METADATA_COLS + available
        if c in df.columns
    ]
    df = df[keep_cols].copy()
    print(f"  {len(available)} features available")

    # impute: max_missing_pct=1.0 so no feature is dropped by threshold
    df = impute_missing(
        df,
        max_missing_pct=1.0,
        length_col="n_tokens_doc" if "n_tokens_doc" in df.columns else None,
        exclude_cols=EXCLUDE_COLS,
        verbose=verbose,
    )

    # percentile capping for numerical stability
    df, _ = cap_extreme_features(
        df,
        exclude_cols=EXCLUDE_COLS,
        verbose=verbose,
    )

    if verbose:
        verify_features(df, exclude_cols=EXCLUDE_COLS, verbose=True)

    return df


# optimization runner


def run_optimization(
    features_csv: str,
    labels_csv: str = None,
    label_col: str = "is_ai",
    id_col: str = "id",
    output_dir: str = None,
    n_iter: int = N_ITER,
    cv_folds: int = CV_FOLDS,
    test_size: float = 0.2,
    scoring: str = "f1",
    random_state: int = 42,
) -> dict:
    """
    Run RandomizedSearchCV over the full XGBoost parameter space.

    args:
        features_csv:  path to stylometric features CSV
        labels_csv:    optional path to CSV with id + label
        label_col:     label column name (default 'is_ai')
        id_col:        document id column (default 'id')
        output_dir:    output directory (default: ./output_optimization/)
        n_iter:        number of random configurations to evaluate (default: 150)
        cv_folds:      stratified CV folds for search (default: 5)
        test_size:     fraction held out for final evaluation (default: 0.2)
        scoring:       CV scoring metric (default: 'f1')
        random_state:  reproducibility seed (default: 42)

    returns:
        dict with best_params, best_cv_score, test_metrics
    """
    if output_dir is None:
        output_dir = str(SCRIPT_DIR / "output_optimization")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    t0 = time.time()

    print("\n" + "=" * 70)
    print(f"XGBOOST RANDOMIZEDSEARCHCV  --  STYLOMETRIC v1.1  (n_iter={n_iter})")
    print("=" * 70)
    print(f"  CV folds:      {cv_folds}")
    print(f"  scoring:       {scoring}")
    print(f"  test split:    {test_size:.0%}")
    print(f"  parameters:    {len(XGB_PARAM_SPACE)} (all XGBoost hyperparameters)")

    # 1. load and clean
    df = load_and_prepare(features_csv, labels_csv, label_col, id_col)

    # 2. build X, y
    feature_cols = [f for f in SELECTED_FEATURES if f in df.columns]
    n_feats = len(feature_cols)
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values

    print(f"\n  samples:       {len(y)}")
    print(f"  features:      {n_feats}")
    print(f"  class balance: {int((y == 1).sum())} AI / {int((y == 0).sum())} human")

    # 3. stratified train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"  train: {len(y_train)}, test: {len(y_test)}")

    # 4. RandomizedSearchCV
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    base_model = xgb.XGBClassifier(
        **XGB_FIXED,
        random_state=random_state,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=XGB_PARAM_SPACE,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        refit=True,
        error_score=np.nan,
    )

    print(f"\n[search] running {n_iter} iterations × {cv_folds}-fold CV...")
    search.fit(X_train, y_train)
    print(f"\n[search] complete: best CV {scoring} = {search.best_score_:.4f}")

    # 5. evaluate best model on held-out test set
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "f1":          f1_score(y_test, y_pred),
        "roc_auc":     roc_auc_score(y_test, y_proba),
        "accuracy":    accuracy_score(y_test, y_pred),
        "precision":   precision_score(y_test, y_pred),
        "recall":      recall_score(y_test, y_pred),
        "best_cv_f1":  search.best_score_,
        "n_iter":      n_iter,
        "cv_folds":    cv_folds,
        "n_features":  n_feats,
        "n_train":     len(y_train),
        "n_test":      len(y_test),
    }

    pd.DataFrame([test_metrics]).to_csv(output_path / "test_metrics.csv", index=False)

    # 6. best params
    best_params_row = {**search.best_params_, "cv_f1": search.best_score_}
    pd.DataFrame([best_params_row]).to_csv(output_path / "best_params.csv", index=False)

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"  best CV {scoring}:    {search.best_score_:.4f}")
    print(f"  test F1:           {test_metrics['f1']:.4f}")
    print(f"  test ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"  test accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  elapsed:           {elapsed / 60:.1f} min")
    print(f"  outputs:           {output_path}")
    print("\nbest parameters:")
    for k, v in sorted(search.best_params_.items()):
        print(f"  {k:<25} {v}")

    return {
        "best_params":   search.best_params_,
        "best_cv_score": search.best_score_,
        "test_metrics":  test_metrics,
    }


# entry point


def main():
    parser = argparse.ArgumentParser(
        description=(
            f"RandomizedSearchCV (n_iter={N_ITER}) for XGBoost "
            f"on stylometric v1.1 features — all hyperparameters"
        )
    )
    parser.add_argument(
        "features_csv",
        nargs="?",
        default=str(DEFAULT_FEATURES_CSV),
        help=f"stylometric features CSV (default: {DEFAULT_FEATURES_CSV})",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        metavar="PATH",
        help="CSV with 'id' + 'is_ai' columns (only if features CSV has no labels)",
    )
    parser.add_argument(
        "--label-col",
        default="is_ai",
        metavar="COL",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "output_optimization"),
        metavar="DIR",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=N_ITER,
        help=f"number of random configurations (default: {N_ITER})",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS,
    )
    parser.add_argument(
        "--scoring",
        default="f1",
        choices=["f1", "roc_auc", "accuracy", "f1_macro"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    run_optimization(
        features_csv=args.features_csv,
        labels_csv=args.labels_csv,
        label_col=args.label_col,
        output_dir=args.output_dir,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
