"""
stratified sampling pipeline.

This module provides functionality to:
1. Load and normalize text data from CSV files
2. Apply text quality gating based on configurable thresholds
3. Assign length bins based on token counts
4. Perform stratified sampling with configurable dimensions

Usage:
    python dataset_sampler.py --config config.yaml

Configuration is provided via YAML file.
"""

import argparse
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml



# Configuration


@dataclass
class TextGateThresholds:
    """Thresholds for text quality gating."""
    min_chars: int = 30
    min_tokens: int = 5
    min_avg_word_length: float = 2.0
    max_avg_word_length: float = 10.0
    min_alpha_ratio: float = 0.55
    max_digit_ratio: float = 0.30
    max_punct_ratio: float = 0.35
    min_entropy_norm: float = 0.35


@dataclass
class StratificationDimension:
    """Configuration for a single stratification dimension."""
    column: str
    weights: Optional[dict[str, float]] = None  # None means proportional
    even: bool = False  # If True, distribute evenly across values
    only_for_label: Optional[Any] = None  # Apply only when label equals this value


@dataclass
class SampleConfig:
    """Configuration for a single output sample."""
    name: str
    size: int
    output_path: str


@dataclass
class LabelDerivation:
    """Configuration for deriving labels from another column."""
    source_column: str
    positive_values: list[str]  # Values in source_column that map to positive class


@dataclass
class Config:
    """Main configuration object."""
    # Input
    input_path: str
    text_column: str
    label_column: str
    
    # Labels
    positive_label: Any = 1  # The label indicating AI-generated (or positive class)
    
    # Optional: derive label from another column
    label_derivation: Optional[LabelDerivation] = None
    
    # Optional: values to exclude from a column before processing
    exclusions: Optional[dict[str, list[str]]] = None
    
    # Optional columns
    id_column: Optional[str] = None
    
    # Pre-sampling (to reduce memory for large datasets)
    presample_size: Optional[int] = None
    
    # Text processing
    strip_markup: bool = True
    normalize_unicode: bool = True
    apply_text_gate: bool = True
    gate_thresholds: TextGateThresholds = field(default_factory=TextGateThresholds)
    
    # Length binning
    length_bin_weights: dict[str, float] = field(
        default_factory=lambda: {"short": 0.25, "medium": 0.25, "long": 0.50}
    )
    
    # Stratification dimensions (beyond the primary label split)
    # Each dimension is applied hierarchically
    stratification: list[StratificationDimension] = field(default_factory=list)
    
    # Output samples
    samples: list[SampleConfig] = field(default_factory=list)
    
    # Reproducibility
    random_seed: int = 422
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        
        # Parse gate thresholds
        gate_thresholds = TextGateThresholds()
        if "gate_thresholds" in raw:
            gate_thresholds = TextGateThresholds(**raw.pop("gate_thresholds"))
        
        # Parse label derivation
        label_derivation = None
        if "label_derivation" in raw:
            ld = raw.pop("label_derivation")
            label_derivation = LabelDerivation(**ld)
        
        # Parse exclusions
        exclusions = raw.pop("exclusions", None)
        
        # Parse stratification dimensions
        strat_dims = []
        if "stratification" in raw:
            for dim in raw.pop("stratification"):
                strat_dims.append(StratificationDimension(**dim))
        
        # Parse sample configs
        samples = []
        if "samples" in raw:
            for s in raw.pop("samples"):
                samples.append(SampleConfig(**s))
        
        # Parse length bin weights
        length_weights = raw.pop(
            "length_bin_weights", 
            {"short": 0.25, "medium": 0.25, "long": 0.50}
        )
        
        return cls(
            gate_thresholds=gate_thresholds,
            label_derivation=label_derivation,
            exclusions=exclusions,
            stratification=strat_dims,
            samples=samples,
            length_bin_weights=length_weights,
            **raw
        )



# Text Normalization and Cleaning


# Regex patterns for markup removal
_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
_HTML_TAG_RE = re.compile(r'<[^>]+>')
_CODE_FENCE_RE = re.compile(r'```[\s\S]*?```', re.M)
_INLINE_CODE_RE = re.compile(r'`[^`]+`')
_MD_TABLE_RE = re.compile(r'^\s*\|.*\|\s*$', re.M)
_WS_COLLAPSE_RE = re.compile(r'\s+')

# Hidden character patterns
_HIDDEN_RE = re.compile(
    r"[\u200B-\u200D\uFEFF\u00AD\u200E\u200F"
    r"\u202A-\u202E\u2060\u2066-\u2069]"
)
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Word tokenizer for diagnostics
_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def strip_markup(text: str) -> tuple[str, dict[str, bool]]:
    """
    Remove URLs, HTML tags, code fences, inline code, and markdown tables.
    
    Returns:
        Tuple of (cleaned_text, flags_dict) where flags indicate what was found.
    """
    if not isinstance(text, str):
        return "", {"had_urls": False, "had_html": False, 
                    "had_code": False, "had_table": False}
    
    flags = {
        "had_urls": bool(_URL_RE.search(text)),
        "had_html": bool(_HTML_TAG_RE.search(text)),
        "had_code": bool(_CODE_FENCE_RE.search(text) or _INLINE_CODE_RE.search(text)),
        "had_table": bool(_MD_TABLE_RE.search(text)),
    }
    
    t = _URL_RE.sub(" ", text)
    t = _CODE_FENCE_RE.sub(" ", t)
    t = _INLINE_CODE_RE.sub(" ", t)
    t = _HTML_TAG_RE.sub(" ", t)
    
    # Remove markdown table rows
    t = "\n".join([ln for ln in t.splitlines() if not _MD_TABLE_RE.match(ln)])
    
    # Collapse whitespace
    t = _WS_COLLAPSE_RE.sub(" ", t).strip()
    
    return t, flags


def normalize_unicode(text: str) -> str:
    """Apply unicode normalization and remove hidden/control characters."""
    if text is None:
        return ""
    
    s = unicodedata.normalize("NFKC", str(text))
    s = _HIDDEN_RE.sub("", s)
    s = _CTRL_RE.sub("", s)
    s = s.replace("\u00A0", " ")  # NBSP to space
    s = re.sub(r"\s+", " ", s).strip()
    
    return s


def normalize_text(text: str, do_strip_markup: bool = True, 
                   do_normalize_unicode: bool = True) -> str:
    """Full text normalization pipeline."""
    if text is None:
        return ""
    
    s = str(text)
    
    if do_strip_markup:
        s, _ = strip_markup(s)
    
    if do_normalize_unicode:
        s = normalize_unicode(s)
    
    return s



# Text Quality Diagnostics and Gating


def compute_text_diagnostics(text: str) -> dict[str, float]:
    """
    Compute diagnostic metrics for a text string.
    
    Returns dictionary with:
        - n_chars: character count
        - n_tok: token count (alphabetic words)
        - alpha_ratio: proportion of alphabetic characters
        - digit_ratio: proportion of digit characters
        - punct_ratio: proportion of punctuation characters
        - avg_word_length: mean word length
        - std_word_length: standard deviation of word length
        - entropy_bits: character-level entropy
        - entropy_norm: normalized entropy (0-1 scale)
    """
    s = text or ""
    n_chars = len(s)
    
    if n_chars == 0:
        return {
            "n_chars": 0, "n_tok": 0,
            "alpha_ratio": 0.0, "digit_ratio": 0.0, "punct_ratio": 0.0,
            "avg_word_length": float("nan"), "std_word_length": float("nan"),
            "entropy_bits": float("nan"), "entropy_norm": float("nan")
        }
    
    n_alpha = sum(c.isalpha() for c in s)
    n_digit = sum(c.isdigit() for c in s)
    n_punct = sum((not c.isalnum()) and (not c.isspace()) for c in s)
    
    tokens = _WORD_RE.findall(s)
    n_tok = len(tokens)
    
    if n_tok > 0:
        avg_wl = sum(len(t) for t in tokens) / n_tok
        std_wl = math.sqrt(sum((len(t) - avg_wl)**2 for t in tokens) / n_tok)
    else:
        avg_wl = float("nan")
        std_wl = float("nan")
    
    # Character-level entropy
    counts = Counter(s)
    probs = [v / n_chars for v in counts.values()]
    entropy = -sum(p * math.log(p, 2) for p in probs)
    entropy_norm = entropy / math.log(max(2, len(counts)), 2)
    
    return {
        "n_chars": n_chars,
        "n_tok": n_tok,
        "alpha_ratio": n_alpha / n_chars,
        "digit_ratio": n_digit / n_chars,
        "punct_ratio": n_punct / n_chars,
        "avg_word_length": avg_wl,
        "std_word_length": std_wl,
        "entropy_bits": entropy,
        "entropy_norm": entropy_norm
    }


def is_text_like(diagnostics: dict[str, float], 
                 thresholds: TextGateThresholds) -> tuple[bool, list[str]]:
    """
    Determine if text meets quality thresholds.
    
    Returns:
        Tuple of (passes_gate, list_of_failure_reasons)
    """
    reasons = []
    
    if diagnostics["n_chars"] < thresholds.min_chars:
        reasons.append("too_short_chars")
    
    if diagnostics["n_tok"] < thresholds.min_tokens:
        reasons.append("too_few_tokens")
    
    awl = diagnostics["avg_word_length"]
    if math.isnan(awl) or awl < thresholds.min_avg_word_length \
            or awl > thresholds.max_avg_word_length:
        reasons.append("avg_word_length_out_of_range")
    
    if diagnostics["alpha_ratio"] < thresholds.min_alpha_ratio:
        reasons.append("low_alpha_ratio")
    
    if diagnostics["digit_ratio"] > thresholds.max_digit_ratio:
        reasons.append("high_digit_ratio")
    
    if diagnostics["punct_ratio"] > thresholds.max_punct_ratio:
        reasons.append("high_punct_ratio")
    
    en = diagnostics["entropy_norm"]
    if math.isnan(en) or en < thresholds.min_entropy_norm:
        reasons.append("low_entropy")
    
    return (len(reasons) == 0, reasons)



# Length Binning


def compute_length_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Add token count and character count columns."""
    out = df.copy()
    out["n_tokens_ws"] = out[text_col].fillna("").map(lambda s: len(str(s).split()))
    out["n_chars_clean"] = out[text_col].fillna("").map(lambda s: len(str(s)))
    return out


def assign_length_bins(df: pd.DataFrame, 
                       weights: dict[str, float],
                       token_col: str = "n_tokens_ws") -> pd.DataFrame:
    """
    Assign length bins based on percentiles derived from weights.
    
    For weights {"short": 0.25, "medium": 0.25, "long": 0.50}:
        - short: <= 25th percentile
        - medium: (25th, 50th]
        - long: > 50th percentile
    """
    out = df.copy()
    
    if len(out) == 0:
        out["length_bin"] = pd.Categorical([], categories=list(weights.keys()))
        return out
    
    # Calculate percentile boundaries from weights
    # Assuming weights are ordered short -> medium -> long
    sorted_bins = sorted(weights.keys(), key=lambda x: weights[x])
    
    # For standard 25/25/50 split: boundaries at 25th and 50th percentile
    cumulative = 0
    boundaries = {}
    for bin_name in sorted_bins[:-1]:  # All but last
        cumulative += weights[bin_name]
        boundaries[bin_name] = np.percentile(out[token_col], cumulative * 100)
    
    def assign_bin(x):
        for bin_name in sorted_bins[:-1]:
            if x <= boundaries[bin_name]:
                return bin_name
        return sorted_bins[-1]  # Last bin
    
    out["length_bin"] = out[token_col].map(assign_bin)
    out["length_bin"] = pd.Categorical(out["length_bin"], categories=list(weights.keys()))
    
    return out



# Allocation Utilities


def largest_remainder(target_total: int, 
                      weights: dict[str, float]) -> dict[str, int]:
    """
    Allocate target_total into integer counts using largest remainder method.
    
    This ensures the sum of allocations exactly equals target_total.
    """
    if not weights:
        return {}
    
    keys = list(weights.keys())
    raw = np.array([weights[k] for k in keys], dtype=float) * target_total
    floors = np.floor(raw).astype(int)
    remainder = target_total - floors.sum()
    
    fracs = raw - floors
    order = np.argsort(-fracs)  # Descending by fractional remainder
    
    alloc = floors.copy()
    for i in range(max(0, remainder)):
        alloc[order[i]] += 1
    
    return {k: int(v) for k, v in zip(keys, alloc)}


def proportional_weights(counts: dict[str, int]) -> dict[str, float]:
    """Convert counts to proportional weights."""
    total = sum(counts.values())
    if total <= 0:
        n = len(counts)
        return {k: 1.0 / n for k in counts} if n else {}
    return {k: v / total for k, v in counts.items()}


def proportional_allocation(target_total: int, 
                            available_counts: dict[str, int]) -> dict[str, int]:
    """
    Allocate target_total proportionally to availability, respecting caps.
    
    Iteratively redistributes when caps bind.
    """
    remaining_total = target_total
    remaining = available_counts.copy()
    alloc = {k: 0 for k in available_counts}
    
    while remaining_total > 0 and any(v > 0 for v in remaining.values()):
        weights = proportional_weights(remaining)
        step = largest_remainder(remaining_total, weights)
        
        # Cap by availability
        for k, v in step.items():
            take = min(v, remaining[k])
            alloc[k] += take
            remaining[k] -= take
        
        remaining_total = target_total - sum(alloc.values())
        
        if not any(v > 0 for v in remaining.values()):
            break
    
    return alloc



# Stratified Sampling


def stratified_sample(
    df: pd.DataFrame,
    total_size: int,
    label_column: str,
    length_weights: dict[str, float],
    stratification: list[StratificationDimension],
    random_seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Create a stratified sample with:
        - 50/50 split on label_column
        - Length bin distribution per length_weights
        - Additional stratification dimensions as configured
    
    Returns:
        Tuple of (sampled_dataframe, diagnostics_dict)
    """
    if len(df) == 0:
        return df.copy(), {"note": "Empty dataframe"}
    
    work = df.dropna(subset=["length_bin"]).copy()
    rng = np.random.default_rng(random_seed)
    
    # Primary split: 50/50 on label
    half = total_size // 2
    remainder = total_size - 2 * half
    
    labels = work[label_column].unique()
    if len(labels) != 2:
        raise ValueError(
            f"Expected binary labels in '{label_column}', found {len(labels)}: {labels}"
        )
    
    label_targets = {labels[0]: half, labels[1]: half + remainder}
    
    # Within each label, allocate to length bins
    length_targets = {
        lab: largest_remainder(target, length_weights) 
        for lab, target in label_targets.items()
    }
    
    picked_indices = []
    diagnostics = {
        "target_total": total_size,
        "label_targets": {str(k): v for k, v in label_targets.items()},
        "length_targets": {str(k): v for k, v in length_targets.items()},
        "by_stratum": {}
    }
    
    for label_val in labels:
        # Filter stratification dimensions for this label
        label_strat = [
            dim for dim in stratification
            if dim.only_for_label is None or dim.only_for_label == label_val
        ]
        
        for lb, lb_target in length_targets[label_val].items():
            subset = work[
                (work[label_column] == label_val) & 
                (work["length_bin"] == lb)
            ]
            
            if len(subset) == 0 or lb_target == 0:
                continue
            
            # Apply additional stratification dimensions
            indices = _sample_with_stratification(
                subset, lb_target, label_strat, rng
            )
            picked_indices.extend(indices)
            
            diagnostics["by_stratum"][(str(label_val), lb)] = len(indices)
    
    sample_df = work.loc[sorted(set(picked_indices))].copy()
    
    diagnostics["picked_total"] = len(sample_df)
    diagnostics["shortfall"] = total_size - len(sample_df)
    
    return sample_df, diagnostics


def _sample_with_stratification(
    df: pd.DataFrame,
    target: int,
    dimensions: list[StratificationDimension],
    rng: np.random.Generator
) -> list:
    """
    Recursively apply stratification dimensions to sample from df.
    
    If no dimensions remain, samples directly from df.
    """
    if len(df) == 0 or target == 0:
        return []
    
    if not dimensions:
        # Base case: sample directly
        take = min(target, len(df))
        return df.sample(n=take, random_state=rng.integers(0, 1_000_000)).index.tolist()
    
    dim = dimensions[0]
    remaining_dims = dimensions[1:]
    
    if dim.column not in df.columns:
        # Skip this dimension if column doesn't exist
        return _sample_with_stratification(df, target, remaining_dims, rng)
    
    # Get available values and their counts
    value_counts = df[dim.column].value_counts().to_dict()
    
    if not value_counts:
        return []
    
    # Determine allocation per value
    if dim.weights is not None:
        # Use specified weights (filter to available values)
        available_weights = {k: v for k, v in dim.weights.items() if k in value_counts}
        if available_weights:
            # Renormalize
            total_w = sum(available_weights.values())
            available_weights = {k: v/total_w for k, v in available_weights.items()}
            alloc = largest_remainder(target, available_weights)
        else:
            alloc = proportional_allocation(target, value_counts)
    elif dim.even:
        # Even distribution across values
        n_vals = len(value_counts)
        even_weights = {k: 1.0/n_vals for k in value_counts}
        alloc = largest_remainder(target, even_weights)
    else:
        # Proportional to availability
        alloc = proportional_allocation(target, value_counts)
    
    # Sample from each stratum
    indices = []
    for val, val_target in alloc.items():
        if val_target <= 0:
            continue
        sub = df[df[dim.column] == val]
        sub_indices = _sample_with_stratification(sub, val_target, remaining_dims, rng)
        indices.extend(sub_indices)
    
    return indices



# Main Pipeline


def load_and_prepare(config: Config) -> pd.DataFrame:
    """Load data and apply preprocessing pipeline."""
    print(f"Loading data from: {config.input_path}")
    
    try:
        df = pd.read_csv(config.input_path)
        print(f"  Initial rows: {len(df):,}")
    except Exception as e:
        print(f"[ERROR] Could not load {config.input_path}: {e}")
        sys.exit(1)
    
    # Validate required columns
    if config.text_column not in df.columns:
        print(f"[ERROR] Text column '{config.text_column}' not found in data")
        sys.exit(1)
    
    # Handle label derivation if configured
    if config.label_derivation:
        ld = config.label_derivation
        if ld.source_column not in df.columns:
            print(f"[ERROR] Label derivation source column '{ld.source_column}' not found")
            sys.exit(1)
        
        print(f"  Deriving label from '{ld.source_column}'...")
        # Normalize source column for matching
        df[ld.source_column] = df[ld.source_column].astype(str).str.strip().str.lower()
        positive_set = set(v.lower() for v in ld.positive_values)
        df[config.label_column] = df[ld.source_column].isin(positive_set)
        print(f"  Label distribution:\n{df[config.label_column].value_counts()}")
    
    if config.label_column not in df.columns:
        print(f"[ERROR] Label column '{config.label_column}' not found in data")
        sys.exit(1)
    
    # Apply exclusions if configured
    if config.exclusions:
        for col, values_to_exclude in config.exclusions.items():
            if col in df.columns:
                # Normalize for matching
                df[col] = df[col].astype(str).str.strip().str.lower()
                exclude_set = set(v.lower() for v in values_to_exclude)
                n_before = len(df)
                df = df[~df[col].isin(exclude_set)].copy()
                print(f"  Excluded {n_before - len(df):,} rows based on '{col}'")
    
    # Pre-sample if configured
    if config.presample_size and len(df) > config.presample_size:
        print(f"  Pre-sampling to {config.presample_size:,} rows...")
        rng = np.random.default_rng(config.random_seed)
        df = df.sample(n=config.presample_size, random_state=rng.integers(0, 1_000_000))
        df.reset_index(drop=True, inplace=True)
    
    # Ensure text column is string and not empty
    df[config.text_column] = df[config.text_column].astype(str)
    df = df[df[config.text_column].str.strip().astype(bool)].copy()
    
    # Deduplicate
    if config.id_column and config.id_column in df.columns:
        df = df.drop_duplicates(subset=[config.id_column])
    else:
        df = df.drop_duplicates(subset=[config.text_column])
    
    print(f"  After dedup: {len(df):,} rows")
    
    # Normalize text
    if config.strip_markup or config.normalize_unicode:
        print("  Normalizing text...")
        df["_text_raw"] = df[config.text_column]
        df[config.text_column] = df[config.text_column].apply(
            lambda x: normalize_text(x, config.strip_markup, config.normalize_unicode)
        )
    
    # Apply text gate
    if config.apply_text_gate:
        print("  Applying text quality gate...")
        diags = df[config.text_column].map(compute_text_diagnostics).apply(pd.Series)
        df = pd.concat([df, diags], axis=1)
        
        decisions = diags.apply(
            lambda row: is_text_like(row.to_dict(), config.gate_thresholds), 
            axis=1
        )
        df["_is_text_like"] = decisions.map(lambda x: x[0])
        df["_gate_fail_reason"] = decisions.map(lambda x: ";".join(x[1]))
        
        n_before = len(df)
        df = df[df["_is_text_like"]].copy()
        print(f"  Removed {n_before - len(df):,} rows failing text gate")
    
    df.reset_index(drop=True, inplace=True)
    print(f"  Final rows: {len(df):,}")
    
    return df


def run_sampling(df: pd.DataFrame, config: Config) -> dict[str, pd.DataFrame]:
    """Run stratified sampling for all configured sample sizes."""
    # Add length features and bins
    print("\nComputing length features...")
    df = compute_length_features(df, config.text_column)
    df = assign_length_bins(df, config.length_bin_weights)
    
    print(f"Length bin distribution:\n{df['length_bin'].value_counts()}\n")
    
    results = {}
    
    for i, sample_cfg in enumerate(config.samples):
        print(f"Creating sample '{sample_cfg.name}' (n={sample_cfg.size:,})...")
        
        sample_df, diagnostics = stratified_sample(
            df=df,
            total_size=sample_cfg.size,
            label_column=config.label_column,
            length_weights=config.length_bin_weights,
            stratification=config.stratification,
            random_seed=config.random_seed + i + 1
        )
        
        print(f"  Sampled: {len(sample_df):,} rows")
        if diagnostics.get("shortfall", 0) > 0:
            print(f"  Warning: shortfall of {diagnostics['shortfall']} rows")
        
        # Save
        output_path = Path(sample_cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(output_path, index=False)
        print(f"  Saved to: {output_path}")
        
        results[sample_cfg.name] = sample_df
    
    return results


def check_balance(df: pd.DataFrame, 
                  expected_total: int,
                  label_column: str,
                  length_weights: dict[str, float]) -> pd.DataFrame:
    """Generate balance validation report."""
    rows = []
    
    if len(df) == 0:
        return pd.DataFrame(columns=["dimension", "category", "actual", "target"])
    
    # Total
    rows.append(("total", "all", len(df), expected_total))
    
    # Label split
    for label, grp in df.groupby(label_column):
        rows.append(("label", str(label), len(grp), expected_total / 2))
    
    # Length bins overall
    for lb, w in length_weights.items():
        target = expected_total * w
        actual = int((df["length_bin"] == lb).sum())
        rows.append(("length_bin", lb, actual, target))
    
    # Length bins within label
    for label, grp in df.groupby(label_column):
        for lb, w in length_weights.items():
            target = (expected_total / 2) * w
            actual = int((grp["length_bin"] == lb).sum())
            rows.append((f"length_bin|{label}", lb, actual, target))
    
    return pd.DataFrame(rows, columns=["dimension", "category", "actual", "target"])



# Entry Point


def main():
    parser = argparse.ArgumentParser(
        description="Data stratified sampling pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run balance validation on output samples"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    config = Config.from_yaml(args.config)
    
    # Run pipeline
    df = load_and_prepare(config)
    results = run_sampling(df, config)
    
    # Validate if requested
    if args.validate:
        print("\nValidating sample balances...")
        
        for sample_cfg in config.samples:
            sample_df = results.get(sample_cfg.name)
            if sample_df is not None:
                print(f"\n{sample_cfg.name} (target={sample_cfg.size}):")
                balance = check_balance(
                    sample_df, 
                    sample_cfg.size,
                    config.label_column,
                    config.length_bin_weights
                )
                print(balance.to_string(index=False))
    
    print("\nDone.")


if __name__ == "__main__":
    main()