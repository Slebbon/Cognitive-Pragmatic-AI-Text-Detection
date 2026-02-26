# 1.1-b_ablation.py
# RFE-based ablation profile for stylometric v1.1 features (27 features)
# sweeps all feature counts 1..N -- no threshold selection, pure profiling


import sys
import os
import argparse
import warnings
import time
import tempfile
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# path setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
UTILS_DIR = PROJECT_ROOT / "Utils"
sys.path.insert(0, str(UTILS_DIR))

from feature_analysis_utils import FeatureAnalyzer
from feature_utils import (
    impute_missing,
    cap_extreme_features,
    verify_features,
)


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

# default features CSV produced by Test_v11_stylometric_extractor.py
DEFAULT_FEATURES_CSV = (
    PROJECT_ROOT
    / "Features-Extractions"
    / "Stylometric"
    / "output"
    / "stylometric_features_27.csv"
)


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
    print(f"  {len(available)} features available for ablation")

    # feature_utils cleaning -------------------------------------------

    # impute with max_missing_pct=1.0 so nothing is dropped by threshold --
    # all NaN are filled, feature selection is left entirely to RFE
    df = impute_missing(
        df,
        max_missing_pct=1.0,
        length_col="n_tokens_doc" if "n_tokens_doc" in df.columns else None,
        exclude_cols=EXCLUDE_COLS,
        verbose=verbose,
    )

    # percentile capping for numerical stability (does not drop features)
    df, _ = cap_extreme_features(
        df,
        exclude_cols=EXCLUDE_COLS,
        verbose=verbose,
    )

    if verbose:
        verify_features(df, exclude_cols=EXCLUDE_COLS, verbose=True)

    return df


# ablation runner


def run_ablation_profile(
    features_csv: str,
    labels_csv: str = None,
    label_col: str = "is_ai",
    id_col: str = "id",
    output_dir: str = None,
    random_state: int = 42,
) -> dict:
    """
    Full RFE ablation profile for stylometric v1.1 features.

    Sweeps all feature counts 1..N using FeatureAnalyzer.run_rfe_ablation().
    No threshold is applied -- results are reported for downstream analysis.

    returns:
        dict with keys: rfe, single_feature
    """
    if output_dir is None:
        output_dir = str(SCRIPT_DIR / "output_ablation")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    t0 = time.time()

    print("\n" + "=" * 70)
    print("RFE ABLATION PROFILE  --  STYLOMETRIC v1.1  (27 features)")
    print("=" * 70)

    # load and clean
    df = load_and_prepare(features_csv, labels_csv, label_col, id_col)

    # save cleaned snapshot
    cleaned_path = output_path / "stylometric_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"\n[prep] cleaned data saved to: {cleaned_path}")

    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    n_feats = len(available_features)

    # FeatureAnalyzer loads from a CSV file, so write a temp file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    try:
        os.close(tmp_fd)
        df.to_csv(tmp_path, index=False)

        analyzer = FeatureAnalyzer(output_dir=output_dir, random_state=random_state)
        analyzer.add_feature_family(
            "stylometric",
            tmp_path,
            feature_columns=available_features,
        )
        analyzer.load_all(id_col=id_col, label_col=label_col)

        # phase 1: single-feature baselines
        print("\n" + "=" * 70)
        print(f"PHASE 1/2  --  SINGLE-FEATURE EVALUATION  ({n_feats} features)")
        print("=" * 70)
        single_df = analyzer.run_single_feature_evaluation()

        # phase 2: RFE full sweep -- 1 to n_feats, no skipping
        steps = list(range(1, n_feats + 1))
        print("\n" + "=" * 70)
        print(f"PHASE 2/2  --  RFE SWEEP  (steps: {steps[0]}..{steps[-1]}, {len(steps)} evaluations)")
        print("=" * 70)
        rfe_df = analyzer.run_rfe_ablation(steps=steps)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # plot
    _plot_rfe_curve(rfe_df, output_path)

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("ABLATION PROFILE COMPLETE")
    print("=" * 70)
    print(f"  features:    {n_feats}")
    print(f"  RFE steps:   {len(steps)} (1 to {n_feats})")
    print(f"  elapsed:     {elapsed / 60:.1f} min")
    print(f"  outputs:     {output_path}")
    _print_rfe_table(rfe_df)

    return {
        "rfe": rfe_df,
        "single_feature": single_df,
    }


# output helpers


def _plot_rfe_curve(rfe_df: pd.DataFrame, output_path: Path):
    """F1 vs number of features with ±1 std band."""
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        rfe_df["n_features"],
        rfe_df["cv_f1_mean"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        color="steelblue",
        label="CV F1 (mean)",
    )
    ax.fill_between(
        rfe_df["n_features"],
        rfe_df["cv_f1_mean"] - rfe_df["cv_f1_std"],
        rfe_df["cv_f1_mean"] + rfe_df["cv_f1_std"],
        alpha=0.2,
        color="steelblue",
        label="±1 std",
    )

    ax.set_xlabel("number of features (RFE, 5-fold CV)")
    ax.set_ylabel("F1 score")
    ax.set_title("RFE Ablation Profile — Stylometric v1.1")
    ax.set_xticks(rfe_df["n_features"])
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = output_path / "rfe_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] RFE curve saved to: {path}")


def _print_rfe_table(rfe_df: pd.DataFrame):
    """Print the full sweep results to stdout."""
    print("\nRFE sweep results (no threshold):")
    print(f"  {'n':>4}  {'CV F1':>8}  {'±std':>6}  selected")
    print("  " + "-" * 70)
    for _, row in rfe_df.iterrows():
        feats = ", ".join(row["selected_features"])
        print(
            f"  {int(row['n_features']):>4}  "
            f"{row['cv_f1_mean']:>8.4f}  "
            f"{row['cv_f1_std']:>6.4f}  "
            f"{feats}"
        )


# entry point


def main():
    parser = argparse.ArgumentParser(
        description="RFE ablation profile for stylometric v1.1 features (27 features, no threshold)"
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
        help="CSV with 'id' + 'is_ai' columns (only needed if features CSV has no labels)",
    )
    parser.add_argument(
        "--label-col",
        default="is_ai",
        metavar="COL",
        help="label column name (default: is_ai)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "output_ablation"),
        metavar="DIR",
        help="output directory (default: ./output_ablation/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    args = parser.parse_args()

    run_ablation_profile(
        features_csv=args.features_csv,
        labels_csv=args.labels_csv,
        label_col=args.label_col,
        output_dir=args.output_dir,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
