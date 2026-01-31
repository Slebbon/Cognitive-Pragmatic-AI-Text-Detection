# feature_utils.py
# utility functions for feature cleaning, correlation analysis, imputation, quality flagging
# designed for stylometric and linguistic feature pipelines, can be reusable for any feature-set; FULLY MODIFICABILE, check for configurations.

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Callable
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# default configuration (can be overridden)

#General boundings for stylometric observed features
# features with natural bounds [0,1] or known ranges
DEFAULT_BOUNDED_FEATURES = {
    # diversity/ratio metrics are bounded [0,1]
    'type_token_ratio': (0.0, 1.0),
    'stopword_ratio': (0.0, 1.0),
    'punctuation_ratio': (0.0, 1.0),
    'uppercase_ratio': (0.0, 1.0),
    'digit_ratio': (0.0, 1.0),
    'whitespace_ratio': (0.0, 1.0),
    'compression_ratio': (0.0, 1.5),
    # sentiment
    'sentiment_polarity': (-1.0, 1.0),
    'sentiment_subjectivity': (0.0, 1.0),
    # readability has practical working range
    'flesch_reading_ease': (-100.0, 150.0),
}

# features where log transform helps stabilize z-scores
DEFAULT_LOG_FEATURES = {'mtld', 'yules_k', 'avg_dependency_distance', 'sentence_length_std', 'max_tree_depth'}


# helper functions


def _finite(s: pd.Series) -> pd.Series:
    """filter to finite values only"""
    return s[np.isfinite(s.values)]

def _quantile(df: pd.DataFrame, col: str, q: float) -> float:
    """safe quantile computation"""
    if col not in df.columns:
        return np.nan
    return float(np.nanquantile(df[col].values, q))

def ensure_numeric(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """coerce columns to numeric, excluding specified columns"""
    df = df.copy()
    exclude = set(exclude_cols or [])
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_numeric_features(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """get numeric feature columns, excluding specified columns"""
    exclude = set(exclude_cols or ['id', 'is_ai', 'label', 'target'])
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

def infer_bounded_features(df: pd.DataFrame, feature_cols: List[str] = None) -> Dict[str, Tuple[float, float]]:
    """
    infer natural bounds for features based on naming patterns and data
    combines defaults with pattern-based detection
    """
    bounds = DEFAULT_BOUNDED_FEATURES.copy()
    
    if feature_cols is None:
        feature_cols = get_numeric_features(df)
    
    for col in feature_cols:
        if col in bounds:
            continue
        
        # pattern-based inference
        col_lower = col.lower()
        
        # ratio/proportion features are [0,1]
        if any(p in col_lower for p in ['ratio', 'proportion', 'rate', 'diversity']):
            bounds[col] = (0.0, 1.0)
        
        # pos ratios
        elif col_lower.startswith('pos_ratio'):
            bounds[col] = (0.0, 1.0)
        
        # entropy is non-negative
        elif 'entropy' in col_lower:
            bounds[col] = (0.0, np.inf)
        
        # per_100 densities
        elif 'per_100' in col_lower:
            bounds[col] = (0.0, 100.0)
    
    return bounds


# correlation analysis


def analyze_correlations(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    threshold: float = 0.85, #can be setted
    method: str = 'pearson',
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    analyze feature correlations and identify highly correlated pairs
    
    args:
        df: input dataframe
        feature_cols: specific columns to analyze (default: all numeric)
        threshold: correlation threshold for flagging pairs
        method: 'pearson', 'spearman', or 'kendall'
        exclude_cols: columns to exclude from analysis
    
    returns:
        corr_matrix: full correlation matrix
        high_corr_pairs: list of (feat1, feat2, correlation) above threshold
    """
    exclude = set(exclude_cols or ['id', 'is_ai', 'label', 'target'])
    
    if feature_cols is None:
        feature_cols = get_numeric_features(df, exclude_cols=exclude)
    else:
        feature_cols = [c for c in feature_cols if c in df.columns and c not in exclude]
    
    if len(feature_cols) < 2:
        print(f"[correlation] insufficient features ({len(feature_cols)})")
        return pd.DataFrame(), []
    
    corr_matrix = df[feature_cols].corr(method=method).abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] > threshold]
        for idx, corr_val in high_corr.items():
            high_corr_pairs.append((str(col), str(idx), float(corr_val)))
    
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return corr_matrix, high_corr_pairs

def suggest_drops_from_correlation(
    df: pd.DataFrame,
    high_corr_pairs: List[Tuple[str, str, float]],
    keep_strategy: str = 'lower_missing',
    priority_features: List[str] = None
) -> Dict[str, str]:
    """
    suggest which features to drop from correlated pairs
    
    args:
        df: input dataframe
        high_corr_pairs: list of (feat1, feat2, correlation)
        keep_strategy: 'lower_missing', 'higher_variance', 'first'
        priority_features: features to always keep if in a correlated pair
    
    returns:
        dict of {feature_to_drop: reason}
    """
    priority = set(priority_features or [])
    to_drop = {}
    already_dropped = set()
    
    for feat1, feat2, corr in high_corr_pairs:
        if feat1 in already_dropped or feat2 in already_dropped:
            continue
        
        if feat1 not in df.columns or feat2 not in df.columns:
            continue
        
        # priority features are always kept
        if feat1 in priority and feat2 not in priority:
            drop_feat, keep_feat = feat2, feat1
            reason = f"corr={corr:.3f} with {keep_feat} (priority feature)"
        elif feat2 in priority and feat1 not in priority:
            drop_feat, keep_feat = feat1, feat2
            reason = f"corr={corr:.3f} with {keep_feat} (priority feature)"
        elif keep_strategy == 'lower_missing':
            miss1 = df[feat1].isna().mean()
            miss2 = df[feat2].isna().mean()
            drop_feat = feat1 if miss1 > miss2 else feat2
            keep_feat = feat2 if miss1 > miss2 else feat1
            reason = f"corr={corr:.3f} with {keep_feat}, missing: {miss1:.1%} vs {miss2:.1%}"
        elif keep_strategy == 'higher_variance':
            var1 = df[feat1].var()
            var2 = df[feat2].var()
            drop_feat = feat1 if var1 < var2 else feat2
            keep_feat = feat2 if var1 < var2 else feat1
            reason = f"corr={corr:.3f} with {keep_feat}, variance: {var1:.4f} vs {var2:.4f}"
        else:  # 'first' - alphabetical
            drop_feat = max(feat1, feat2)
            keep_feat = min(feat1, feat2)
            reason = f"corr={corr:.3f} with {keep_feat}, alphabetical"
        
        to_drop[drop_feat] = reason
        already_dropped.add(drop_feat)
    
    return to_drop

def drop_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.85,
    keep_strategy: str = 'lower_missing',
    priority_features: List[str] = None,
    exclude_cols: List[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    analyze and drop highly correlated features
    
    returns:
        df_filtered: dataframe with correlated features removed
        drop_log: dict of {dropped_feature: reason}
    """
    if verbose:
        print(f"\n[correlation] analyzing with threshold={threshold}")
    
    _, high_corr_pairs = analyze_correlations(df, threshold=threshold, exclude_cols=exclude_cols)
    
    if not high_corr_pairs:
        if verbose:
            print(f"[correlation] no pairs above threshold")
        return df, {}
    
    if verbose:
        print(f"[correlation] found {len(high_corr_pairs)} highly correlated pairs")
    
    drop_log = suggest_drops_from_correlation(df, high_corr_pairs, keep_strategy, priority_features)
    
    if verbose:
        print(f"[correlation] suggesting {len(drop_log)} features for removal:")
        for feat, reason in drop_log.items():
            print(f"  - {feat}: {reason}")
    
    df_filtered = df.drop(columns=list(drop_log.keys()), errors='ignore')
    
    if verbose:
        print(f"[correlation] features: {len(df.columns)} -> {len(df_filtered.columns)}")
    
    return df_filtered, drop_log


# custom feature dropping, not suggested.


def drop_features_by_config(
    df: pd.DataFrame,
    drop_config: Dict[str, str],
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    drop features based on a configuration dict
    
    args:
        df: input dataframe
        drop_config: dict of {feature_name: reason_for_dropping}
    
    returns:
        df_filtered: dataframe with features removed
        drop_log: dict of actually dropped features
    """
    existing_drops = {k: v for k, v in drop_config.items() if k in df.columns}
    
    if verbose and existing_drops:
        print(f"\n[config drop] removing {len(existing_drops)} features:")
        for feat, reason in existing_drops.items():
            print(f"  - {feat}: {reason}")
    
    df_filtered = df.drop(columns=list(existing_drops.keys()), errors='ignore')
    
    return df_filtered, existing_drops

def drop_features_by_pattern(
    df: pd.DataFrame,
    patterns: List[str],
    reason: str = "pattern match",
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    drop features matching regex patterns
    
    args:
        df: input dataframe
        patterns: list of regex patterns to match
        reason: reason string for the drop log
    
    returns:
        df_filtered: dataframe with matched features removed
        drop_log: dict of dropped features
    """
    import re
    
    drop_log = {}
    for col in df.columns:
        for pattern in patterns:
            if re.search(pattern, col, re.IGNORECASE):
                drop_log[col] = f"{reason} ({pattern})"
                break
    
    if verbose and drop_log:
        print(f"\n[pattern drop] removing {len(drop_log)} features:")
        for feat, r in drop_log.items():
            print(f"  - {feat}: {r}")
    
    df_filtered = df.drop(columns=list(drop_log.keys()), errors='ignore')
    
    return df_filtered, drop_log


# percentile-based capping


def calculate_caps(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    bounded_features: Dict[str, Tuple[float, float]] = None,
    exclude_cols: List[str] = None
) -> Dict[str, Tuple[float, float]]:
    """
    calculate percentile-based caps respecting natural bounds
    
    args:
        df: input dataframe
        feature_cols: specific features to cap
        lower_pct: lower percentile (default 1)
        upper_pct: upper percentile (default 99)
        bounded_features: dict of {feature: (min, max)} natural bounds
        exclude_cols: columns to exclude
    
    returns:
        dict of {feature: (lower_cap, upper_cap)}
    """
    exclude = set(exclude_cols or ['id', 'is_ai', 'label', 'target'])
    
    if feature_cols is None:
        feature_cols = get_numeric_features(df, exclude_cols=exclude)
    
    # combine default bounds with inferred and provided
    bounds = infer_bounded_features(df, feature_cols)
    if bounded_features:
        bounds.update(bounded_features)
    
    caps = {}
    
    for feat in feature_cols:
        if feat not in df.columns or feat in exclude:
            continue
        
        s = _finite(df[feat].dropna())
        if s.empty:
            continue
        
        lo = float(np.percentile(s, lower_pct))
        hi = float(np.percentile(s, upper_pct))
        
        # apply natural bounds
        if feat in bounds:
            blo, bhi = bounds[feat]
            lo = max(lo, blo)
            hi = min(hi, bhi) if not np.isinf(bhi) else hi
        
        if lo > hi:
            lo, hi = hi, lo
        
        caps[feat] = (lo, hi)
    
    return caps

def apply_caps(
    df: pd.DataFrame,
    caps: Dict[str, Tuple[float, float]],
    verbose: bool = True
) -> pd.DataFrame:
    """
    apply percentile-based capping to features
    
    returns:
        df with capped values
    """
    df = df.copy()
    cap_counts = {}
    
    for feat, (lo, hi) in caps.items():
        if feat not in df.columns:
            continue
        
        before = df[feat].copy()
        df[feat] = df[feat].clip(lower=lo, upper=hi)
        
        changed = (before != df[feat]) & df[feat].notna() & before.notna()
        n_changed = int(changed.sum())
        
        if n_changed > 0:
            cap_counts[feat] = n_changed
    
    if verbose and cap_counts:
        print(f"\n[capping] capped values in {len(cap_counts)} features:")
        for feat, count in sorted(cap_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {feat}: {count} values ({100*count/len(df):.1f}%)")
        if len(cap_counts) > 10:
            print(f"  ... and {len(cap_counts) - 10} more")
    
    return df

def cap_extreme_features(
    df: pd.DataFrame,
    lower_pct: float = 1.0, #can be changed, settable
    upper_pct: float = 99.0, #can be changed, settable
    bounded_features: Dict[str, Tuple[float, float]] = None,
    exclude_cols: List[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    convenience function: calculate and apply caps in one step
    """
    caps = calculate_caps(df, lower_pct=lower_pct, upper_pct=upper_pct, 
                          bounded_features=bounded_features, exclude_cols=exclude_cols)
    
    if verbose:
        print(f"\n[capping] calculated caps for {len(caps)} features (p{lower_pct}-p{upper_pct})")
    
    df_capped = apply_caps(df, caps, verbose=verbose)
    
    return df_capped, caps


# quality flagging and diagnostics


def compute_diagnostic_thresholds(
    df: pd.DataFrame,
    feature_quantiles: Dict[str, Tuple[str, float]] = None
) -> Dict[str, float]:
    """
    compute data-driven thresholds for quality diagnostics
    
    args:
        df: input dataframe
        feature_quantiles: dict of {threshold_name: (feature_name, quantile)}
    
    returns:
        dict of {threshold_name: value}
    """
    # default thresholds for common stylometric features based on thesis observed heuristics, CAN be changed.
    defaults = {
        'asl_hi': ('avg_sentence_length', 0.99),
        'sls_hi': ('sentence_length_std', 0.99),
        'depth_max_hi': ('max_tree_depth', 0.995),
        'depth_avg_hi': ('avg_tree_depth', 0.995),
        'depdist_hi': ('avg_dependency_distance', 0.995),
        'comp_hi_fixed': (None, 1.0),  # fixed value
        'upper_hi': ('uppercase_ratio', 0.999),
        'uniq_hi': ('unique_char_count', 0.999),
        'ws_lo': ('whitespace_ratio', 0.005),
        'ws_hi': ('whitespace_ratio', 0.999),
    }
    
    if feature_quantiles:
        defaults.update(feature_quantiles)
    
    thresholds = {}
    for name, (feat, q) in defaults.items():
        if feat is None:
            thresholds[name] = q  # fixed value
        else:
            thresholds[name] = _quantile(df, feat, q)
    
    return thresholds

def diagnose_outliers(
    df: pd.DataFrame,
    custom_checks: Dict[str, Callable[[pd.DataFrame], pd.Series]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    diagnose feature outliers with customizable checks
    
    args:
        df: input dataframe
        custom_checks: dict of {flag_name: function(df) -> bool_series}
    
    returns:
        diagnostics dataframe with flags and severity scores
    """
    df = ensure_numeric(df, exclude_cols=['id'])
    thr = compute_diagnostic_thresholds(df)
    
    D = pd.DataFrame(index=df.index)
    
    # default checks for common issues
    default_checks = {}
    
    if 'avg_sentence_length' in df.columns:
        default_checks['seg_len_extreme'] = lambda d: (
            (d.get('avg_sentence_length', np.nan) > thr.get('asl_hi', np.inf)) |
            (d.get('sentence_length_std', np.nan) > thr.get('sls_hi', np.inf))
        ).fillna(False)
    
    if 'max_tree_depth' in df.columns:
        default_checks['depth_implausible'] = lambda d: (
            (d.get('max_tree_depth', np.nan) > thr.get('depth_max_hi', np.inf)) |
            (d.get('avg_tree_depth', np.nan) > thr.get('depth_avg_hi', np.inf))
        ).fillna(False)
    
    if 'avg_dependency_distance' in df.columns:
        default_checks['dep_distance_implausible'] = lambda d: (
            d.get('avg_dependency_distance', np.nan) > thr.get('depdist_hi', np.inf)
        ).fillna(False)
    
    if 'compression_ratio' in df.columns:
        default_checks['compression_anomaly'] = lambda d: (
            d.get('compression_ratio', np.nan) > thr.get('comp_hi_fixed', 1.0)
        ).fillna(False)
    
    if 'uppercase_ratio' in df.columns:
        default_checks['char_anomaly'] = lambda d: (
            (d.get('uppercase_ratio', np.nan) > thr.get('upper_hi', np.inf)) |
            (d.get('unique_char_count', np.nan) > thr.get('uniq_hi', np.inf))
        ).fillna(False)
    
    # apply default checks
    for name, check_fn in default_checks.items():
        try:
            D[name] = check_fn(df)
        except Exception:
            D[name] = False
    
    # apply custom checks
    if custom_checks:
        for name, check_fn in custom_checks.items():
            try:
                D[name] = check_fn(df)
            except Exception:
                D[name] = False
    
    # severity score
    flag_cols = [c for c in D.columns]
    D['severity'] = D[flag_cols].sum(axis=1).astype(int)
    
    if verbose:
        print(f"\n[diagnostics] outlier summary:")
        for col in flag_cols:
            count = D[col].sum()
            if count > 0:
                print(f"  - {col}: {count} ({100*count/len(D):.1f}%)")
        print(f"  - total with any issue: {(D['severity'] > 0).sum()}")
    
    D.attrs['thresholds'] = thr
    return D

def add_quality_flags(
    df: pd.DataFrame,
    check_features: Dict[str, List[str]] = None,
    n_std: float = 3.0,
    log_features: Set[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    add statistical quality flags based on z-scores
    
    args:
        df: input dataframe
        check_features: dict of {flag_name: [features_to_check]}
        n_std: number of standard deviations for threshold
        log_features: features to log-transform before z-score
    
    returns:
        df with quality flag columns added
    """
    df = df.copy()
    log_feats = log_features or DEFAULT_LOG_FEATURES
    
    def zscore_on(series: pd.Series, log_transform: bool) -> pd.Series:
        s = series.astype(float).replace([np.inf, -np.inf], np.nan)
        if log_transform:
            nonan = s.dropna()
            if len(nonan) > 0 and (nonan >= 0).all():
                s = np.log1p(s)
        m, sd = s.mean(), s.std(ddof=0)
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return ((s - m) / sd).abs()
    
    # default feature groups if not provided
    if check_features is None:
        check_features = {
            'parse_quality_issue': ['avg_dependency_distance', 'max_tree_depth', 'avg_tree_depth'],
            'readability_anomaly': ['flesch_reading_ease', 'gunning_fog', 'automated_readability_index'],
            'lexical_anomaly': ['mtld', 'yules_k', 'type_token_ratio'],
        }
    
    # compute z-scores and flags
    for flag_name, features in check_features.items():
        flag_series = pd.Series(False, index=df.index)
        
        for feat in features:
            if feat in df.columns:
                z = zscore_on(df[feat], log_transform=(feat in log_feats))
                flag_series = flag_series | (z > n_std)
        
        # also flag missing values for parse features
        if 'parse' in flag_name.lower():
            for feat in features:
                if feat in df.columns:
                    flag_series = flag_series | df[feat].isna()
        
        df[flag_name] = flag_series.fillna(False)
    
    # compute quality score
    flag_cols = list(check_features.keys())
    existing_flags = [c for c in flag_cols if c in df.columns]
    df['quality_score'] = (len(existing_flags) - df[existing_flags].sum(axis=1)).clip(lower=0)
    
    if verbose:
        print(f"\n[quality] flags added (threshold: {n_std} std):")
        for col in existing_flags:
            print(f"  - {col}: {df[col].sum()}")
        print(f"  - quality score distribution:\n{df['quality_score'].value_counts().sort_index().to_string()}")
    
    return df


# imputation


def impute_missing(
    df: pd.DataFrame,
    max_missing_pct: float = 0.2, #Set as you want <---- 
    group_col: str = None,
    length_col: str = 'n_tokens_doc',
    length_bins: List[float] = None,
    exclude_cols: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    impute missing values with stratified group medians
    
    args:
        df: input dataframe
        max_missing_pct: drop features with more missing than this
        group_col: column to group by for stratified imputation
        length_col: column to use for length-based binning (if group_col not provided)
        length_bins: bin edges for length-based grouping
        exclude_cols: columns to exclude from imputation
    
    returns:
        df with imputed values
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # remove duplicate columns
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].unique().tolist()
        if verbose:
            print(f"[impute] removing duplicate columns: {dups}")
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    exclude = set(exclude_cols or ['id', 'is_ai', 'label', 'target'])
    num_feats = get_numeric_features(df, exclude_cols=exclude)
    
    if not num_feats:
        return df
    
    # step 1: drop high-missing features
    missing_pct = df[num_feats].isna().mean()
    high_missing = missing_pct[missing_pct > max_missing_pct].sort_values(ascending=False)
    
    if not high_missing.empty:
        if verbose:
            print(f"\n[impute] dropping {len(high_missing)} features with >{max_missing_pct*100:.0f}% missing:")
            for feat, pct in high_missing.items():
                print(f"  - {feat}: {pct*100:.1f}%")
        num_feats = [f for f in num_feats if f not in high_missing.index]
        df.drop(columns=high_missing.index, inplace=True)
    
    if not num_feats:
        return df
    
    # step 2: determine grouping
    group_keys = []
    
    if group_col and group_col in df.columns:
        group_keys = [group_col]
    elif length_col and length_col in df.columns:
        bins = length_bins or [0, 100, 250, 500, 10000]
        labels = [f'bin_{i}' for i in range(len(bins)-1)]
        df['__len_bin__'] = pd.cut(df[length_col], bins=bins, right=False, labels=labels)
        group_keys = ['__len_bin__']
    
    before_missing = df[num_feats].isna().sum()
    
    # step 3: stratified imputation
    if group_keys:
        for feat in num_feats:
            if df[feat].isna().any():
                group_medians = df.groupby(group_keys, dropna=False, observed=False)[feat].transform('median')
                df[feat] = df[feat].fillna(group_medians)
    
    # step 4: global median fallback
    filled_counts = {}
    for feat in num_feats:
        if df[feat].isna().any():
            median_val = df[feat].median()
            n_filled = df[feat].isna().sum()
            df[feat] = df[feat].fillna(median_val)
            if n_filled > 0:
                filled_counts[feat] = (n_filled, median_val)
    
    if verbose and filled_counts:
        print(f"\n[impute] global median fallback:")
        for feat, (n, med) in filled_counts.items():
            print(f"  - {feat}: {n} values filled with {med:.4f}")
    
    df.drop(columns=['__len_bin__'], errors='ignore', inplace=True)
    
    # summary
    after_missing = df[num_feats].isna().sum()
    total_imputed = int((before_missing - after_missing).sum())
    
    if verbose and total_imputed > 0:
        print(f"\n[impute] total values imputed: {total_imputed}")
    
    return df


# full cleaning pipeline


def clean_features_pipeline(
    df: pd.DataFrame,
    custom_drops: Dict[str, str] = None,
    drop_correlated: bool = True,
    corr_threshold: float = 0.85,
    corr_keep_strategy: str = 'lower_missing',
    priority_features: List[str] = None,
    cap_extremes: bool = True,
    cap_lower_pct: float = 1.0,
    cap_upper_pct: float = 99.0,
    bounded_features: Dict[str, Tuple[float, float]] = None,
    add_quality: bool = True,
    quality_n_std: float = 3.0,
    quality_check_features: Dict[str, List[str]] = None,
    impute: bool = True,
    max_missing_pct: float = 0.4,
    exclude_cols: List[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    full feature cleaning pipeline with configurable steps
    
    steps:
    1. custom drops (user-defined redundancies)
    2. correlation-based drops
    3. percentile capping
    4. quality flagging
    5. imputation
    
    returns:
        df_clean: cleaned dataframe
        metadata: dict with all logs and parameters
    """
    if verbose:
        print("="*70)
        print("FEATURE CLEANING PIPELINE")
        print("="*70)
    
    metadata = {'original_shape': df.shape}
    exclude = list(exclude_cols or ['id', 'is_ai', 'label', 'target'])
    
    # step 1: custom drops
    if custom_drops:
        df, custom_log = drop_features_by_config(df, custom_drops, verbose=verbose)
        metadata['custom_drops'] = custom_log
    
    # step 2: correlation drops
    if drop_correlated:
        df, corr_log = drop_correlated_features(
            df, threshold=corr_threshold, keep_strategy=corr_keep_strategy,
            priority_features=priority_features, exclude_cols=exclude, verbose=verbose
        )
        metadata['correlation_drops'] = corr_log
    
    # step 3: capping
    if cap_extremes:
        df, caps = cap_extreme_features(
            df, lower_pct=cap_lower_pct, upper_pct=cap_upper_pct,
            bounded_features=bounded_features, exclude_cols=exclude, verbose=verbose
        )
        metadata['caps'] = caps
    
    # step 4: quality flags
    if add_quality:
        df = add_quality_flags(
            df, check_features=quality_check_features, n_std=quality_n_std, verbose=verbose
        )
    
    # step 5: imputation
    if impute:
        df = impute_missing(df, max_missing_pct=max_missing_pct, exclude_cols=exclude, verbose=verbose)
    
    metadata['final_shape'] = df.shape
    
    if verbose:
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print(f"  original: {metadata['original_shape']}")
        print(f"  final: {metadata['final_shape']}")
        print("="*70)
    
    return df, metadata


# verification and reporting


def verify_features(df: pd.DataFrame, exclude_cols: List[str] = None, verbose: bool = True) -> Dict:
    """
    verify feature quality after cleaning
    
    returns:
        dict with verification results
    """
    exclude = set(exclude_cols or ['id', 'is_ai', 'label', 'target'])
    results = {}
    
    # nan check
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    results['nan_columns'] = nan_cols.to_dict()
    
    # inf check
    num_cols = get_numeric_features(df, exclude_cols=exclude)
    inf_counts = {}
    for col in num_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            inf_counts[col] = n_inf
    results['inf_columns'] = inf_counts
    
    # value ranges
    ranges = {}
    for col in num_cols:
        ranges[col] = {'min': df[col].min(), 'max': df[col].max(), 'mean': df[col].mean()}
    results['ranges'] = ranges
    
    # summary stats
    results['n_features'] = len(num_cols)
    results['n_samples'] = len(df)
    
    if verbose:
        print("\n[verification] results:")
        print(f"  - features: {results['n_features']}")
        print(f"  - samples: {results['n_samples']}")
        print(f"  - columns with nan: {len(nan_cols)}")
        print(f"  - columns with inf: {len(inf_counts)}")
        
        if nan_cols.any():
            print("\n  nan details:")
            for col, count in list(nan_cols.items())[:10]:
                print(f"    {col}: {count} ({100*count/len(df):.1f}%)")
            if len(nan_cols) > 10:
                print(f"    ... and {len(nan_cols) - 10} more")
    
    return results

def generate_cleaning_report(metadata: Dict, output_path: str = None) -> str:
    """generate text report from cleaning metadata"""
    lines = [
        "FEATURE CLEANING REPORT",
        "="*60,
        f"Original shape: {metadata.get('original_shape', 'N/A')}",
        f"Final shape: {metadata.get('final_shape', 'N/A')}",
        ""
    ]
    
    if 'custom_drops' in metadata and metadata['custom_drops']:
        lines.append(f"Custom drops ({len(metadata['custom_drops'])}):")
        for feat, reason in metadata['custom_drops'].items():
            lines.append(f"  - {feat}: {reason}")
        lines.append("")
    
    if 'correlation_drops' in metadata and metadata['correlation_drops']:
        lines.append(f"Correlation drops ({len(metadata['correlation_drops'])}):")
        for feat, reason in metadata['correlation_drops'].items():
            lines.append(f"  - {feat}: {reason}")
        lines.append("")
    
    if 'caps' in metadata and metadata['caps']:
        lines.append(f"Capped features ({len(metadata['caps'])}):")
        for feat, (lo, hi) in list(metadata['caps'].items())[:15]:
            lines.append(f"  - {feat}: [{lo:.4f}, {hi:.4f}]")
        if len(metadata['caps']) > 15:
            lines.append(f"  ... and {len(metadata['caps']) - 15} more")
        lines.append("")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return report


# convenience functions


def quick_clean(
    df: pd.DataFrame,
    exclude_cols: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    quick cleaning with sensible defaults
    
    usage:
        from feature_utils import quick_clean
        df_clean = quick_clean(df)
    """
    df_clean, _ = clean_features_pipeline(
        df,
        drop_correlated=True,
        corr_threshold=0.85,
        cap_extremes=True,
        add_quality=True,
        impute=True,
        exclude_cols=exclude_cols,
        verbose=verbose
    )
    return df_clean

def merge_feature_sets(
    dfs: List[pd.DataFrame],
    on: str = 'id',
    how: str = 'outer',
    verbose: bool = True
) -> pd.DataFrame:
    """
    merge multiple feature dataframes
    
    usage:
        df_all = merge_feature_sets([df_stylometric, df_coherence, df_cognitive])
    """
    if not dfs:
        return pd.DataFrame()
    
    result = dfs[0].copy()
    
    for i, df in enumerate(dfs[1:], 1):
        # avoid duplicate columns (except merge key)
        new_cols = [c for c in df.columns if c not in result.columns or c == on]
        result = result.merge(df[new_cols], on=on, how=how)
        
        if verbose:
            print(f"[merge] added df {i}: {len(new_cols)-1} new features")
    
    if verbose:
        print(f"[merge] final shape: {result.shape}")
    
    return result