"""
Feature Analysis Utilities
Comprehensive evaluation pipeline for AI text detection features
Supports multiple feature families with flexible combinations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
import pickle

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import RFE
from scipy import stats
import xgboost as xgb

warnings.filterwarnings('ignore')


# Data structures

@dataclass
class FeatureFamily:
    """Container for a feature family"""
    name: str
    csv_path: str
    feature_columns: List[str] = None  # If None, auto-detect
    
    def __post_init__(self):
        self.df = None
        self.loaded = False

@dataclass 
class AnalysisResults:
    """Container for analysis results"""
    family_name: str
    cv_f1_mean: float
    cv_f1_std: float
    test_f1: float
    test_roc_auc: float
    feature_importance: pd.DataFrame
    statistical_tests: pd.DataFrame


# Analyzer class

class FeatureAnalyzer:
    """
    Comprehensive feature analysis for AI text detection
    
    Usage:
        analyzer = FeatureAnalyzer(output_dir="results")
        analyzer.add_feature_family("backbone", "backbone.csv")
        analyzer.add_feature_family("perplexity", "perplexity.csv")
        analyzer.load_all()
        analyzer.run_full_analysis()
    """
    
    def __init__(self, output_dir: str = "analysis_results", random_state: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        
        self.feature_families: Dict[str, FeatureFamily] = {}
        self.merged_df: pd.DataFrame = None
        self.results: Dict[str, AnalysisResults] = {}
        
        # XGBoost config
        self.xgb_params = {
            'n_estimators': 500,
            'max_depth': 9,
            'learning_rate': 0.15,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    
    
    def add_feature_family(self, name: str, csv_path: str, feature_columns: List[str] = None):
        """Add a feature family to analyze"""
        self.feature_families[name] = FeatureFamily(name, csv_path, feature_columns)
        print(f"Added feature family: {name}")
    
    def load_all(self, id_col: str = 'id', label_col: str = 'is_ai', labels_csv: str = None):
        """
        Load all feature families and merge them
        
        Args:
            id_col: Column name for document ID (must exist in all CSVs)
            label_col: Column name for labels (is_ai)
            labels_csv: Optional separate CSV containing id and labels. 
                       If None, labels are expected in at least one feature CSV.
                       This is useful when feature CSVs only contain id + features.
        """
        print("\n" + "=" * 70)
        print("LOADING FEATURE FAMILIES")
        print("=" * 70)
        
        # Standard columns to exclude from features
        exclude_cols = {id_col, label_col, 'text', 'generation', 'model', 'domain', 
                       'source', 'prompt', 'attack', 'language'}
        
        # Track all feature columns across families to detect overlaps
        all_feature_cols = {}
        dfs = []
        labels_df = None
        
        # Load labels from separate file if provided
        if labels_csv is not None:
            print(f"\nLoading labels from: {labels_csv}")
            labels_df = pd.read_csv(labels_csv)
            if id_col not in labels_df.columns or label_col not in labels_df.columns:
                raise ValueError(f"Labels CSV must contain '{id_col}' and '{label_col}' columns")
            labels_df = labels_df[[id_col, label_col]].drop_duplicates()
            print(f"  Loaded labels for {len(labels_df)} documents")
        
        # Load each feature family
        for name, family in self.feature_families.items():
            print(f"\nLoading {name} from {family.csv_path}...")
            df = pd.read_csv(family.csv_path)
            
            # Check for id column
            if id_col not in df.columns:
                raise ValueError(f"ID column '{id_col}' not found in {name}")
            
            # Auto-detect feature columns if not specified
            if family.feature_columns is None:
                # Exclude standard columns and get only numeric
                candidate_cols = [c for c in df.columns if c not in exclude_cols]
                family.feature_columns = [c for c in candidate_cols 
                                         if df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
            
            # Check for overlapping feature names with other families
            for col in family.feature_columns:
                if col in all_feature_cols:
                    print(f"  WARNING: '{col}' also exists in {all_feature_cols[col]}, will be renamed")
                    # Rename to avoid conflicts
                    new_name = f"{col}_{name}"
                    df = df.rename(columns={col: new_name})
                    family.feature_columns = [new_name if c == col else c for c in family.feature_columns]
                else:
                    all_feature_cols[col] = name
            
            # Check if this CSV has labels (for backwards compatibility)
            has_labels = label_col in df.columns
            
            family.df = df
            family.loaded = True
            print(f"  Loaded {len(df)} rows, {len(family.feature_columns)} features")
            print(f"  Has labels: {has_labels}")
            if len(family.feature_columns) <= 5:
                print(f"  Features: {family.feature_columns}")
            else:
                print(f"  Features: {family.feature_columns[:5]}... (+{len(family.feature_columns)-5} more)")
            
            dfs.append((name, df, has_labels))
        
        # Merge all dataframes
        print("\n" + "-" * 50)
        print("MERGING FEATURE FAMILIES")
        print("-" * 50)
        
        # Start with labels if provided separately, otherwise use first df
        if labels_df is not None:
            self.merged_df = labels_df.copy()
            print(f"Starting with labels: {len(self.merged_df)} rows")
        else:
            # Find first df that has labels
            base_df = None
            for name, df, has_labels in dfs:
                if has_labels:
                    base_df = df[[id_col, label_col]].drop_duplicates()
                    print(f"Using labels from: {name}")
                    break
            
            if base_df is None:
                raise ValueError(f"No CSV contains '{label_col}' column. Either add labels to a feature CSV or provide labels_csv parameter.")
            
            self.merged_df = base_df.copy()
        
        # Merge each feature family (only id + feature columns)
        for name, df, has_labels in dfs:
            family = self.feature_families[name]
            cols_to_merge = [id_col] + family.feature_columns
            cols_to_merge = [c for c in cols_to_merge if c in df.columns]
            
            before_merge = len(self.merged_df)
            self.merged_df = self.merged_df.merge(df[cols_to_merge], on=id_col, how='inner')
            after_merge = len(self.merged_df)
            
            print(f"  + {name}: {len(family.feature_columns)} features, {after_merge} rows (lost {before_merge - after_merge})")
        
        print(f"\nFinal merged dataset: {len(self.merged_df)} samples")
        
        # Store config
        self.id_col = id_col
        self.label_col = label_col
        
        # Verify and report
        if label_col not in self.merged_df.columns:
            raise ValueError(f"Label column '{label_col}' not found after merge")
        
        n_ai = (self.merged_df[label_col] == 1).sum()
        n_human = (self.merged_df[label_col] == 0).sum()
        total_features = sum(len(f.feature_columns) for f in self.feature_families.values())
        
        print(f"\n  Total features: {total_features}")
        print(f"  AI samples: {n_ai} ({100*n_ai/len(self.merged_df):.1f}%)")
        print(f"  Human samples: {n_human} ({100*n_human/len(self.merged_df):.1f}%)")
        
        # List all features by family
        print(f"\n  Features by family:")
        for name, family in self.feature_families.items():
            print(f"    {name}: {len(family.feature_columns)} features")
    
    def get_feature_columns(self, families: List[str] = None) -> List[str]:
        """Get feature columns for specified families"""
        if families is None:
            families = list(self.feature_families.keys())
        
        cols = []
        for name in families:
            if name in self.feature_families:
                cols.extend(self.feature_families[name].feature_columns)
        return cols
    
    
    # Stats analysis
    
    def run_statistical_tests(self, families: List[str] = None) -> pd.DataFrame:
        """Run t-tests and compute Cohen's d for all features"""
        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("=" * 70)
        
        feature_cols = self.get_feature_columns(families)
        ai_df = self.merged_df[self.merged_df[self.label_col] == 1]
        human_df = self.merged_df[self.merged_df[self.label_col] == 0]
        
        results = []
        for feat in feature_cols:
            ai_vals = ai_df[feat].dropna()
            human_vals = human_df[feat].dropna()
            
            # T-test
            t_stat, p_val = stats.ttest_ind(ai_vals, human_vals)
            
            # Cohen's d
            pooled_std = np.sqrt(((len(ai_vals)-1)*ai_vals.var() + (len(human_vals)-1)*human_vals.var()) / (len(ai_vals)+len(human_vals)-2))
            cohens_d = (ai_vals.mean() - human_vals.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Effect size interpretation
            abs_d = abs(cohens_d)
            effect = "large" if abs_d >= 0.8 else "medium" if abs_d >= 0.5 else "small" if abs_d >= 0.2 else "negligible"
            
            results.append({
                'feature': feat,
                'ai_mean': ai_vals.mean(),
                'human_mean': human_vals.mean(),
                'diff': ai_vals.mean() - human_vals.mean(),
                'diff_pct': 100 * (ai_vals.mean() - human_vals.mean()) / abs(human_vals.mean()) if human_vals.mean() != 0 else 0,
                't_statistic': t_stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'effect_size': effect,
                'significant': p_val < 0.05
            })
        
        results_df = pd.DataFrame(results).sort_values('p_value')
        
        # Summary
        sig_count = results_df['significant'].sum()
        bonf_alpha = 0.05 / len(feature_cols)
        bonf_sig = (results_df['p_value'] < bonf_alpha).sum()
        
        print(f"\nSignificant at p<0.05: {sig_count}/{len(feature_cols)}")
        print(f"Bonferroni-corrected (α={bonf_alpha:.4f}): {bonf_sig}/{len(feature_cols)}")
        print(f"\nEffect sizes: Large={sum(results_df['effect_size']=='large')}, Medium={sum(results_df['effect_size']=='medium')}, Small={sum(results_df['effect_size']=='small')}")
        
        # Save
        path = self.output_dir / "statistical_tests.csv"
        results_df.to_csv(path, index=False)
        print(f"Saved to: {path}")
        
        return results_df
    
    
    # Training configurations with AUTONOMOUS train_test split.
    
    def train_and_evaluate(self, families: List[str] = None, name: str = None) -> Dict:
        """Train XGBoost and evaluate with cross-validation"""
        feature_cols = self.get_feature_columns(families)
        if name is None:
            name = "+".join(families) if families else "all"
        
        print(f"\n{'='*70}")
        print(f"TRAINING: {name} ({len(feature_cols)} features)")
        print("="*70)
        
        # Prepare data
        X = self.merged_df[feature_cols].fillna(self.merged_df[feature_cols].median())
        y = self.merged_df[self.label_col].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        
        # Cross-validation
        clf = xgb.XGBClassifier(**self.xgb_params)
        cv_scores = cross_val_score(clf, X, y, cv=self.cv, scoring='f1')
        
        # Train final model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        test_f1 = f1_score(y_test, y_pred)
        test_roc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Test ROC-AUC: {test_roc:.4f}")
        
        return {
            'name': name,
            'n_features': len(feature_cols),
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'test_f1': test_f1,
            'test_roc_auc': test_roc,
            'model': clf,
            'feature_cols': feature_cols
        }
    
    
    # Feature Importance
    
    def compute_xgb_importance(self, families: List[str] = None) -> pd.DataFrame:
        """Compute XGBoost gain importance"""
        print("\n" + "=" * 70)
        print("XGB GAIN IMPORTANCE")
        print("=" * 70)
        
        feature_cols = self.get_feature_columns(families)
        X = self.merged_df[feature_cols].fillna(self.merged_df[feature_cols].median())
        y = self.merged_df[self.label_col].values
        
        clf = xgb.XGBClassifier(**self.xgb_params)
        clf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'gain': clf.feature_importances_,
            'gain_pct': 100 * clf.feature_importances_ / clf.feature_importances_.sum()
        }).sort_values('gain', ascending=False)
        
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        print("\nTop 15 features by XGB Gain:")
        print("-" * 60)
        for _, row in importance_df.head(15).iterrows():
            print(f"  {row['rank']:2d}. {row['feature']:<40} {row['gain_pct']:.2f}%")
        
        path = self.output_dir / "xgb_gain_importance.csv"
        importance_df.to_csv(path, index=False)
        print(f"\nSaved to: {path}")
        
        return importance_df
    
    def compute_shap_importance(self, families: List[str] = None) -> pd.DataFrame:
        """Compute SHAP importance"""
        print("\n" + "=" * 70)
        print("SHAP IMPORTANCE")
        print("=" * 70)
        
        try:
            import shap
        except ImportError:
            print("SHAP not installed. Run: pip install shap")
            return None
        
        feature_cols = self.get_feature_columns(families)
        X = self.merged_df[feature_cols].fillna(self.merged_df[feature_cols].median())
        y = self.merged_df[self.label_col].values
        
        # Use subset for SHAP (faster)
        X_sample = X.sample(n=min(1000, len(X)), random_state=self.random_state)
        
        clf = xgb.XGBClassifier(**self.xgb_params)
        clf.fit(X, y)
        
        print("Computing SHAP values (this may take a few minutes)...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sample)
        
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': feature_cols,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        shap_df['rank'] = range(1, len(shap_df) + 1)
        
        print("\nTop 15 features by SHAP:")
        print("-" * 60)
        for _, row in shap_df.head(15).iterrows():
            print(f"  {row['rank']:2d}. {row['feature']:<40} {row['shap_importance']:.4f}")
        
        path = self.output_dir / "shap_importance.csv"
        shap_df.to_csv(path, index=False)
        print(f"\nSaved to: {path}")
        
        # Save SHAP plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_df
    
    #Opzionale <--
    def compute_permutation_importance(self, families: List[str] = None) -> pd.DataFrame:
        """Compute permutation importance"""
        print("\n" + "=" * 70)
        print("PERMUTATION IMPORTANCE")
        print("=" * 70)
        
        from sklearn.inspection import permutation_importance
        
        feature_cols = self.get_feature_columns(families)
        X = self.merged_df[feature_cols].fillna(self.merged_df[feature_cols].median())
        y = self.merged_df[self.label_col].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y)
        
        clf = xgb.XGBClassifier(**self.xgb_params)
        clf.fit(X_train, y_train)
        
        print("Computing permutation importance...")
        perm = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=self.random_state, scoring='f1')
        
        perm_df = pd.DataFrame({
            'feature': feature_cols,
            'perm_importance': perm.importances_mean,
            'perm_std': perm.importances_std
        }).sort_values('perm_importance', ascending=False)
        perm_df['rank'] = range(1, len(perm_df) + 1)
        
        print("\nTop 15 features by Permutation Importance:")
        print("-" * 60)
        for _, row in perm_df.head(15).iterrows():
            print(f"  {row['rank']:2d}. {row['feature']:<40} {row['perm_importance']:.4f}")
        
        path = self.output_dir / "permutation_importance.csv"
        perm_df.to_csv(path, index=False)
        print(f"\nSaved to: {path}")
        
        return perm_df
    
    
    # Ablation studys
    
    def run_rfe_ablation(self, families: List[str] = None, steps: List[int] = None) -> pd.DataFrame:
        """Run Recursive Feature Elimination ablation"""
        print("\n" + "=" * 70)
        print("RFE ABLATION STUDY")
        print("=" * 70)
        
        feature_cols = self.get_feature_columns(families)
        X = self.merged_df[feature_cols].fillna(self.merged_df[feature_cols].median())
        y = self.merged_df[self.label_col].values
        
        if steps is None:
            steps = [1, 2, 3, 5, 7, 10, 15, 20, len(feature_cols)]
            steps = [s for s in steps if s <= len(feature_cols)]
        
        results = []
        for n_features in steps:
            print(f"\nTesting with {n_features} features...")
            
            # RFE to select features
            clf = xgb.XGBClassifier(**self.xgb_params)
            rfe = RFE(clf, n_features_to_select=n_features, step=1)
            rfe.fit(X, y)
            
            selected_features = [f for f, s in zip(feature_cols, rfe.support_) if s]
            X_selected = X[selected_features]
            
            # Cross-validation
            cv_scores = cross_val_score(xgb.XGBClassifier(**self.xgb_params), X_selected, y, cv=self.cv, scoring='f1')
            
            results.append({
                'n_features': n_features,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'selected_features': selected_features
            })
            
            print(f"  F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        results_df = pd.DataFrame(results)
        
        # Find optimal
        best_idx = results_df['cv_f1_mean'].idxmax()
        best = results_df.loc[best_idx]
        print(f"\nOptimal: {best['n_features']} features, F1={best['cv_f1_mean']:.4f}")
        
        path = self.output_dir / "rfe_ablation.csv"
        results_df.to_csv(path, index=False)
        print(f"Saved to: {path}")
        
        return results_df
    
    def run_single_feature_evaluation(self, families: List[str] = None) -> pd.DataFrame:
        """Evaluate each feature individually"""
        print("\n" + "=" * 70)
        print("SINGLE FEATURE EVALUATION")
        print("=" * 70)
        
        feature_cols = self.get_feature_columns(families)
        y = self.merged_df[self.label_col].values
        
        results = []
        for feat in feature_cols:
            X_single = self.merged_df[[feat]].fillna(self.merged_df[feat].median())
            
            clf = xgb.XGBClassifier(**self.xgb_params)
            cv_scores = cross_val_score(clf, X_single, y, cv=self.cv, scoring='f1')
            
            results.append({
                'feature': feat,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std()
            })
        
        results_df = pd.DataFrame(results).sort_values('cv_f1_mean', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        
        print("\nTop 15 individual features:")
        print("-" * 70)
        for _, row in results_df.head(15).iterrows():
            print(f"  {row['rank']:2d}. {row['feature']:<40} F1={row['cv_f1_mean']:.4f}")
        
        path = self.output_dir / "single_feature_evaluation.csv"
        results_df.to_csv(path, index=False)
        print(f"\nSaved to: {path}")
        
        return results_df
    
    #OPZIONALE
    
    def run_incremental_ablation(self, families: List[str] = None, baseline_family: str = None) -> pd.DataFrame:
        """Add features incrementally to baseline"""
        print("\n" + "=" * 70)
        print("INCREMENTAL ABLATION STUDY")
        print("=" * 70)
        
        if baseline_family is None:
            baseline_family = list(self.feature_families.keys())[0]
        
        baseline_cols = self.feature_families[baseline_family].feature_columns
        other_families = [f for f in self.feature_families.keys() if f != baseline_family]
        
        y = self.merged_df[self.label_col].values
        results = []
        
        # Baseline only
        X_base = self.merged_df[baseline_cols].fillna(self.merged_df[baseline_cols].median())
        cv_scores = cross_val_score(xgb.XGBClassifier(**self.xgb_params), X_base, y, cv=self.cv, scoring='f1')
        baseline_f1 = cv_scores.mean()
        
        results.append({
            'configuration': f"{baseline_family} only",
            'n_features': len(baseline_cols),
            'cv_f1_mean': baseline_f1,
            'cv_f1_std': cv_scores.std(),
            'delta_f1': 0.0,
            'delta_pct': 0.0
        })
        print(f"\n{baseline_family} only: F1={baseline_f1:.4f}")
        
        # Add each family incrementally
        current_cols = baseline_cols.copy()
        for family in other_families:
            family_cols = self.feature_families[family].feature_columns
            current_cols = current_cols + family_cols
            
            X_combo = self.merged_df[current_cols].fillna(self.merged_df[current_cols].median())
            cv_scores = cross_val_score(xgb.XGBClassifier(**self.xgb_params), X_combo, y, cv=self.cv, scoring='f1')
            
            delta = cv_scores.mean() - baseline_f1
            results.append({
                'configuration': f"{baseline_family} + {family}",
                'n_features': len(current_cols),
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'delta_f1': delta,
                'delta_pct': 100 * delta / baseline_f1
            })
            print(f"+ {family}: F1={cv_scores.mean():.4f} (Δ={delta:+.4f})")
        
        results_df = pd.DataFrame(results)
        path = self.output_dir / "incremental_ablation.csv"
        results_df.to_csv(path, index=False)
        print(f"\nSaved to: {path}")
        
        return results_df
    
    
    #Opzionale-> comparazione famiglie
    
    def compare_all_families(self) -> pd.DataFrame:
        """Compare performance of each family individually and combined"""
        print("\n" + "=" * 70)
        print("FAMILY COMPARISON")
        print("=" * 70)
        
        results = []
        
        # Each family individually
        for name in self.feature_families.keys():
            res = self.train_and_evaluate([name], name)
            results.append({
                'configuration': name,
                'n_features': res['n_features'],
                'cv_f1_mean': res['cv_f1_mean'],
                'cv_f1_std': res['cv_f1_std'],
                'test_f1': res['test_f1'],
                'test_roc_auc': res['test_roc_auc']
            })
        
        # All combined
        if len(self.feature_families) > 1:
            res = self.train_and_evaluate(list(self.feature_families.keys()), "all_combined")
            results.append({
                'configuration': 'all_combined',
                'n_features': res['n_features'],
                'cv_f1_mean': res['cv_f1_mean'],
                'cv_f1_std': res['cv_f1_std'],
                'test_f1': res['test_f1'],
                'test_roc_auc': res['test_roc_auc']
            })
        
        results_df = pd.DataFrame(results)
        
        print("\n" + "=" * 90)
        print(f"{'Configuration':<25} {'Features':<10} {'CV F1':<20} {'Test F1':<12} {'ROC-AUC':<12}")
        print("=" * 90)
        for _, row in results_df.iterrows():
            print(f"{row['configuration']:<25} {row['n_features']:<10} {row['cv_f1_mean']:.4f}±{row['cv_f1_std']:.4f}     {row['test_f1']:<12.4f} {row['test_roc_auc']:<12.4f}")
        
        path = self.output_dir / "family_comparison.csv"
        results_df.to_csv(path, index=False)
        print(f"\nSaved to: {path}")
        
        return results_df
    
    
    # Opzionale plots, non settati di default >
    
    def plot_family_comparison(self, results_df: pd.DataFrame):
        """Plot family comparison bar chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(results_df))
        colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))
        
        bars = ax.bar(x, results_df['cv_f1_mean'], yerr=results_df['cv_f1_std'], 
                      capsize=5, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['configuration'], rotation=45, ha='right')
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_title('Feature Family Performance Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, results_df['cv_f1_mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "family_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {self.output_dir / 'family_comparison.png'}")
    
    def plot_importance_comparison(self, gain_df: pd.DataFrame, shap_df: pd.DataFrame = None, perm_df: pd.DataFrame = None):
        """Plot importance rankings comparison"""
        fig, axes = plt.subplots(1, 3 if shap_df is not None and perm_df is not None else 1, figsize=(15, 8))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # XGB Gain
        ax = axes[0]
        top_n = min(15, len(gain_df))
        y_pos = range(top_n)
        ax.barh(y_pos, gain_df.head(top_n)['gain_pct'], color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(gain_df.head(top_n)['feature'], fontsize=9)
        ax.set_xlabel('Importance (%)')
        ax.set_title('XGB Gain Importance', fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "importance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    
    # Analysis pipeline example
    
    def run_full_analysis(self, include_shap: bool = True):
        """Run complete analysis pipeline"""
        print("\n" + "=" * 70)
        print("RUNNING FULL ANALYSIS PIPELINE")
        print("=" * 70)
        
        # 1. Statistical tests
        stat_df = self.run_statistical_tests()
        
        # 2. Family comparison
        comparison_df = self.compare_all_families()
        self.plot_family_comparison(comparison_df)
        
        # 3. Feature importance
        gain_df = self.compute_xgb_importance()
        perm_df = self.compute_permutation_importance()
        
        shap_df = None
        if include_shap:
            shap_df = self.compute_shap_importance()
        
        self.plot_importance_comparison(gain_df, shap_df, perm_df)
        
        # 4. Single feature evaluation
        single_df = self.run_single_feature_evaluation()
        
        # 5. RFE ablation
        rfe_df = self.run_rfe_ablation()
        
        # 6. Incremental ablation
        if len(self.feature_families) > 1:
            incr_df = self.run_incremental_ablation()
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nAll results saved to: {self.output_dir}")
        
        return {
            'statistical_tests': stat_df,
            'family_comparison': comparison_df,
            'xgb_importance': gain_df,
            'permutation_importance': perm_df,
            'shap_importance': shap_df,
            'single_feature': single_df,
            'rfe_ablation': rfe_df
        }



#Altre funzioni

def quick_analysis(feature_csvs: Dict[str, str], output_dir: str = "analysis_results", 
                   include_shap: bool = True, labels_csv: str = None):
    """
    Quick analysis with minimal setup
    
    Args:
        feature_csvs: Dict mapping family name to CSV path
                     e.g., {"backbone": "backbone.csv", "perplexity": "perplexity.csv"}
        output_dir: Where to save results
        include_shap: Whether to compute SHAP values (slower)
        labels_csv: Optional path to CSV with 'id' and 'is_ai' columns.
                   Use this if your feature CSVs don't contain labels.
    
    Example:
        # If each feature CSV has id + features only (no is_ai):
        analyzer, results = quick_analysis(
            {"backbone": "backbone.csv", "perplexity": "perplexity.csv"},
            labels_csv="original_dataset.csv"  # Contains id and is_ai
        )
        
        # If at least one CSV has is_ai column:
        analyzer, results = quick_analysis(
            {"backbone": "backbone_with_labels.csv", "perplexity": "perplexity.csv"}
        )
    """
    analyzer = FeatureAnalyzer(output_dir=output_dir)
    
    for name, path in feature_csvs.items():
        analyzer.add_feature_family(name, path)
    
    analyzer.load_all(labels_csv=labels_csv)
    results = analyzer.run_full_analysis(include_shap=include_shap)
    
    return analyzer, results



# Main

def main():
    print("FEATURE ANALYSIS UTILITY - INTERACTIVE MODE")
    
    analyzer = FeatureAnalyzer()
    
    # Get feature families
    print("\nEnter feature families (empty line to finish):")
    while True:
        name = input("  Family name (e.g., 'backbone'): ").strip()
        if not name:
            break
        path = input(f"  CSV path for {name}: ").strip()
        analyzer.add_feature_family(name, path)
    
    if not analyzer.feature_families:
        print("No feature families added. Exiting.")
        return
    
    # Load data
    analyzer.load_all()
    
    # Run analysis
    print("\nSelect analyses to run:")
    print("  1. Full analysis (all)")
    print("  2. Statistical tests only")
    print("  3. Family comparison only")
    print("  4. Feature importance only")
    print("  5. Ablation studies only")
    
    choice = input("\nChoice [1]: ").strip() or "1"
    
    if choice == "1":
        include_shap = input("Include SHAP? [y/n]: ").strip().lower() == 'y'
        analyzer.run_full_analysis(include_shap=include_shap)
    elif choice == "2":
        analyzer.run_statistical_tests()
    elif choice == "3":
        analyzer.compare_all_families()
    elif choice == "4":
        analyzer.compute_xgb_importance()
        analyzer.compute_permutation_importance()
    elif choice == "5":
        analyzer.run_single_feature_evaluation()
        analyzer.run_rfe_ablation()
    
    print(f"\nResults saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()