"""
Hyperparameter Search Utilities
Configurable optimization with learning curves and plateau detection.
"""

import numpy as np
import pandas as pd
import yaml
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

from sklearn.model_selection import (
    StratifiedKFold, train_test_split, learning_curve,
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, make_scorer
)
from scipy.stats import wilcoxon, uniform, randint, loguniform
from scipy.ndimage import uniform_filter1d

#IMPORT MODELS 

#try:
#    from xgboost import XGBClassifier
#    HAS_XGBOOST = True
#except ImportError:
#    HAS_XGBOOST = False


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SearchResult:
    best_params: Dict
    best_score: float
    best_score_std: float
    all_results: List[Dict]
    best_estimator: Any
    search_time_seconds: float
    n_configurations_tested: int


@dataclass
class LearningCurveResult:
    train_sizes: np.ndarray
    train_scores: Dict[str, np.ndarray]
    val_scores: Dict[str, np.ndarray]
    plateau_detected: bool
    plateau_index: Optional[int]
    analysis_time_seconds: float


@dataclass
class PlateauAnalysis:
    detected: bool
    index: Optional[int]
    method_used: str
    details: Dict


class ConfigParser:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _validate_config(self):
        required = ['global', 'data', 'cross_validation', 'search', 'model', 'param_space']
        for section in required:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
    
    def get(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class ParameterSampler:
    @staticmethod
    def sample_value(param_config: Dict, rng: np.random.RandomState) -> Any:
        param_type = param_config.get('type', 'discrete')
        
        if param_type == 'discrete':
            return rng.choice(param_config['values'])
        elif param_type == 'int_range':
            return int(rng.randint(param_config['min'], param_config['max'] + 1))
        elif param_type == 'uniform':
            return rng.uniform(param_config['min'], param_config['max'])
        elif param_type == 'log_uniform':
            log_min = np.log(param_config['min'])
            log_max = np.log(param_config['max'])
            return float(np.exp(rng.uniform(log_min, log_max)))
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    @classmethod
    def sample_configuration(cls, param_space: Dict, rng: np.random.RandomState) -> Dict:
        return {name: cls.sample_value(cfg, rng) for name, cfg in param_space.items()}
    
    @classmethod
    def generate_grid(cls, param_space: Dict) -> List[Dict]:
        import itertools
        
        param_lists = {}
        for name, cfg in param_space.items():
            ptype = cfg.get('type', 'discrete')
            if ptype == 'discrete':
                param_lists[name] = cfg['values']
            elif ptype == 'int_range':
                param_lists[name] = list(range(cfg['min'], cfg['max'] + 1))
            else:
                param_lists[name] = np.linspace(cfg['min'], cfg['max'], 5).tolist()
        
        keys = list(param_lists.keys())
        combinations = list(itertools.product(*[param_lists[k] for k in keys]))
        return [dict(zip(keys, combo)) for combo in combinations]
    
    @classmethod
    def to_sklearn_distributions(cls, param_space: Dict) -> Dict:
        """Convert param_space to sklearn-compatible distributions for RandomizedSearchCV."""
        distributions = {}
        for name, cfg in param_space.items():
            ptype = cfg.get('type', 'discrete')
            if ptype == 'discrete':
                distributions[name] = cfg['values']
            elif ptype == 'int_range':
                distributions[name] = randint(cfg['min'], cfg['max'] + 1)
            elif ptype == 'uniform':
                distributions[name] = uniform(cfg['min'], cfg['max'] - cfg['min'])
            elif ptype == 'log_uniform':
                distributions[name] = loguniform(cfg['min'], cfg['max'])
        return distributions


class ModelRegistry:
    _models: Dict[str, Tuple[type, Dict]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type, default_params: Optional[Dict] = None):
        cls._models[name] = (model_class, default_params or {})
    
    @classmethod
    def get(cls, name: str) -> Tuple[type, Dict]:
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def create(cls, name: str, params: Dict) -> Any:
        model_class, defaults = cls.get(name)
        merged = {**defaults, **params}
        return model_class(**merged)
    
    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._models.keys())


if HAS_XGBOOST:
    ModelRegistry.register('xgboost', XGBClassifier, {'n_jobs': -1, 'eval_metric': 'logloss', 'verbosity': 0})

ModelRegistry.register('random_forest', RandomForestClassifier, {'n_jobs': -1})
ModelRegistry.register('gradient_boosting', GradientBoostingClassifier, {})

if HAS_LIGHTGBM:
    ModelRegistry.register('lightgbm', LGBMClassifier, {'n_jobs': -1, 'verbose': -1})

ModelRegistry.register('svm', SVC, {'probability': True})
ModelRegistry.register('logistic_regression', LogisticRegression, {'max_iter': 1000})


class CrossValidationSearch:
    def __init__(
        self,
        model_name: str,
        param_space: Dict,
        fixed_params: Dict,
        cv: StratifiedKFold,
        scoring: str = 'f1',
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: Optional[int] = None
    ):
        self.model_name = model_name
        self.param_space = param_space
        self.fixed_params = fixed_params
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self._scoring_funcs = {
            'f1': f1_score,
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
        }
    
    def _evaluate_config(self, params: Dict, X: np.ndarray, y: np.ndarray) -> Optional[Dict]:
        merged = {**self.fixed_params, **params}
        if self.random_state is not None:
            merged['random_state'] = self.random_state
        merged = self._sanitize_params(merged)
        
        scores = []
        for train_idx, val_idx in self.cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                model = ModelRegistry.create(self.model_name, merged)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                if self.scoring == 'roc_auc':
                    y_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_proba)
                else:
                    score = self._scoring_funcs[self.scoring](y_val, y_pred)
                scores.append(score)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"    Config failed: {e}")
                return None
        
        return {'params': params, 'mean_score': np.mean(scores), 'std_score': np.std(scores), 'scores': scores}
    
    def _sanitize_params(self, params: Dict) -> Dict:
        params = params.copy()
        
        if self.model_name == 'logistic_regression':
            penalty = params.get('penalty', 'l2')
            solver = params.get('solver', 'lbfgs')
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                params['solver'] = 'saga'
            if penalty == 'elasticnet' and solver != 'saga':
                params['solver'] = 'saga'
            if penalty != 'elasticnet':
                params.pop('l1_ratio', None)
        
        if self.model_name == 'svm':
            if params.get('kernel') != 'poly':
                params.pop('degree', None)
        
        return params
    
    def search_random(self, X: np.ndarray, y: np.ndarray, n_iter: int = 30,
                      checkpoint_callback: Optional[Callable] = None) -> List[Dict]:
        results = []
        for i in range(n_iter):
            params = ParameterSampler.sample_configuration(self.param_space, self.rng)
            if self.verbose >= 1:
                print(f"  [{i+1}/{n_iter}] Testing configuration...")
            
            result = self._evaluate_config(params, X, y)
            if result is not None:
                results.append(result)
                if self.verbose >= 1:
                    print(f"    Score: {result['mean_score']:.4f} +/- {result['std_score']:.4f}")
                if checkpoint_callback:
                    checkpoint_callback(results)
        return results
    
    def search_grid(self, X: np.ndarray, y: np.ndarray,
                    checkpoint_callback: Optional[Callable] = None) -> List[Dict]:
        grid = ParameterSampler.generate_grid(self.param_space)
        results = []
        
        if self.verbose >= 1:
            print(f"  Grid search: {len(grid)} configurations")
        
        for i, params in enumerate(grid):
            if self.verbose >= 1:
                print(f"  [{i+1}/{len(grid)}] Testing configuration...")
            
            result = self._evaluate_config(params, X, y)
            if result is not None:
                results.append(result)
                if self.verbose >= 1:
                    print(f"    Score: {result['mean_score']:.4f} +/- {result['std_score']:.4f}")
                if checkpoint_callback:
                    checkpoint_callback(results)
        return results


class SklearnRandomizedSearch:
    """Wrapper around sklearn's RandomizedSearchCV."""
    
    def __init__(
        self,
        model_name: str,
        param_space: Dict,
        fixed_params: Dict,
        cv: StratifiedKFold,
        scoring: str = 'f1',
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: Optional[int] = None
    ):
        self.model_name = model_name
        self.param_space = param_space
        self.fixed_params = fixed_params
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
    
    def search(self, X: np.ndarray, y: np.ndarray, n_iter: int = 30) -> List[Dict]:
        base_params = self.fixed_params.copy()
        if self.random_state is not None:
            base_params['random_state'] = self.random_state
        
        base_model = ModelRegistry.create(self.model_name, base_params)
        distributions = ParameterSampler.to_sklearn_distributions(self.param_space)
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=distributions,
            n_iter=n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            return_train_score=False
        )
        
        search.fit(X, y)
        
        results = []
        for i in range(len(search.cv_results_['params'])):
            results.append({
                'params': search.cv_results_['params'][i],
                'mean_score': search.cv_results_['mean_test_score'][i],
                'std_score': search.cv_results_['std_test_score'][i],
                'scores': []
            })
        
        return results


class PlateauDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.method = config.get('method', 'gradient')
    
    def detect(self, scores: np.ndarray) -> PlateauAnalysis:
        if self.method == 'gradient':
            return self._gradient_method(scores)
        elif self.method == 'threshold':
            return self._threshold_method(scores)
        elif self.method == 'statistical':
            return self._statistical_method(scores)
        raise ValueError(f"Unknown plateau method: {self.method}")
    
    def _gradient_method(self, scores: np.ndarray) -> PlateauAnalysis:
        cfg = self.config.get('gradient', {})
        window = cfg.get('window_size', 3)
        min_imp = cfg.get('min_improvement', 0.001)
        consec = cfg.get('consecutive_checks', 2)
        
        smoothed = uniform_filter1d(scores, size=window, mode='nearest')
        improvements = np.diff(smoothed) / (np.abs(smoothed[:-1]) + 1e-10)
        
        plateau_idx = None
        consecutive_count = 0
        for i, below in enumerate(improvements < min_imp):
            if below:
                consecutive_count += 1
                if consecutive_count >= consec:
                    plateau_idx = i - consec + 2
                    break
            else:
                consecutive_count = 0
        
        return PlateauAnalysis(detected=plateau_idx is not None, index=plateau_idx,
                               method_used='gradient', details={'improvements': improvements.tolist()})
    
    def _threshold_method(self, scores: np.ndarray) -> PlateauAnalysis:
        cfg = self.config.get('threshold', {})
        target = cfg.get('target_score', 0.95)
        tol = cfg.get('tolerance', 0.005)
        
        reached_idx = None
        for i, score in enumerate(scores):
            if abs(score - target) <= tol or score >= target:
                reached_idx = i
                break
        
        return PlateauAnalysis(detected=reached_idx is not None, index=reached_idx,
                               method_used='threshold', details={'target': target})
    
    def _statistical_method(self, scores: np.ndarray) -> PlateauAnalysis:
        cfg = self.config.get('statistical', {})
        alpha = cfg.get('significance_level', 0.05)
        min_samples = cfg.get('min_samples', 5)
        
        if len(scores) < min_samples * 2:
            return PlateauAnalysis(detected=False, index=None, method_used='statistical',
                                   details={'error': 'Insufficient samples'})
        
        mid = len(scores) // 2
        first_half = scores[:mid]
        second_half = scores[mid:]
        
        try:
            stat, p_value = wilcoxon(second_half[:len(first_half)] - first_half)
            detected = p_value > alpha
        except Exception:
            detected, p_value = False, None
        
        return PlateauAnalysis(detected=detected, index=mid if detected else None,
                               method_used='statistical', details={'p_value': p_value})


class LearningCurveAnalyzer:
    def __init__(self, config: Dict, plateau_config: Dict):
        self.config = config
        self.plateau_detector = PlateauDetector(plateau_config)
    
    def compute(self, estimator: Any, X: np.ndarray, y: np.ndarray,
                random_state: Optional[int] = None) -> LearningCurveResult:
        start_time = time.time()
        train_sizes = self._build_train_sizes(len(X))
        
        cv = StratifiedKFold(
            n_splits=self.config.get('cv_splits', 5),
            shuffle=self.config.get('shuffle', True),
            random_state=random_state
        )
        
        scoring_list = self.config.get('scoring', ['f1'])
        if isinstance(scoring_list, str):
            scoring_list = [scoring_list]
        
        train_scores_dict, val_scores_dict = {}, {}
        
        for scoring in scoring_list:
            train_sizes_out, train_scores, val_scores = learning_curve(
                estimator, X, y, train_sizes=train_sizes, cv=cv,
                scoring=scoring, n_jobs=-1, random_state=random_state
            )
            train_scores_dict[scoring] = train_scores
            val_scores_dict[scoring] = val_scores
        
        val_means = val_scores_dict[scoring_list[0]].mean(axis=1)
        plateau_result = self.plateau_detector.detect(val_means)
        
        return LearningCurveResult(
            train_sizes=train_sizes_out,
            train_scores=train_scores_dict,
            val_scores=val_scores_dict,
            plateau_detected=plateau_result.detected,
            plateau_index=plateau_result.index,
            analysis_time_seconds=time.time() - start_time
        )
    
    def _build_train_sizes(self, n_samples: int) -> np.ndarray:
        ts_cfg = self.config.get('train_sizes', {})
        ts_type = ts_cfg.get('type', 'linspace')
        
        if ts_type == 'linspace':
            return np.linspace(ts_cfg.get('start', 0.1), ts_cfg.get('stop', 1.0), ts_cfg.get('num', 10))
        elif ts_type == 'logspace':
            return np.logspace(np.log10(ts_cfg.get('start', 0.1)), np.log10(ts_cfg.get('stop', 1.0)), ts_cfg.get('num', 10))
        elif ts_type == 'list':
            return np.array(ts_cfg.get('values', [0.1, 0.3, 0.5, 0.7, 1.0]))
        return np.linspace(0.1, 1.0, 10)


class Visualizer:
    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style('whitegrid')
        style_cfg = config.get('style', {})
        plt.rcParams['figure.dpi'] = style_cfg.get('figure_dpi', 300)
        plt.rcParams['font.size'] = style_cfg.get('font_size', 10)
    
    def plot_learning_curves(self, result: LearningCurveResult, model_name: str = "Model") -> plt.Figure:
        lc_cfg = self.config.get('learning_curve', {}).get('plot', {})
        figsize = lc_cfg.get('figsize', [12, 5])
        show_std = lc_cfg.get('show_std', True)
        alpha_fill = lc_cfg.get('alpha_fill', 0.2)
        
        n_metrics = len(result.train_scores)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, result.train_scores.keys()):
            train_mean = result.train_scores[metric].mean(axis=1)
            train_std = result.train_scores[metric].std(axis=1)
            val_mean = result.val_scores[metric].mean(axis=1)
            val_std = result.val_scores[metric].std(axis=1)
            
            ax.plot(result.train_sizes, train_mean, 'o-', label='Training', color='steelblue')
            ax.plot(result.train_sizes, val_mean, 'o-', label='Validation', color='darkorange')
            
            if show_std:
                ax.fill_between(result.train_sizes, train_mean - train_std, train_mean + train_std,
                               alpha=alpha_fill, color='steelblue')
                ax.fill_between(result.train_sizes, val_mean - val_std, val_mean + val_std,
                               alpha=alpha_fill, color='darkorange')
            
            if result.plateau_detected and result.plateau_index is not None:
                ax.axvline(result.train_sizes[result.plateau_index], color='red',
                          linestyle='--', alpha=0.7, label=f'Plateau')
            
            ax.set_xlabel('Training Set Size', fontweight='bold')
            ax.set_ylabel(metric.upper(), fontweight='bold')
            ax.set_title(f'{model_name} Learning Curve ({metric})', fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / f'learning_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        return fig
    
    def plot_search_results(self, results: List[Dict], top_n: int = 10) -> plt.Figure:
        sorted_results = sorted(results, key=lambda x: x['mean_score'], reverse=True)[:top_n]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        means = [r['mean_score'] for r in sorted_results]
        stds = [r['std_score'] for r in sorted_results]
        axes[0].barh(range(len(sorted_results)), means, xerr=stds, capsize=4, color='steelblue', alpha=0.7)
        axes[0].set_yticks(range(len(sorted_results)))
        axes[0].set_yticklabels([f"Config {i+1}" for i in range(len(sorted_results))])
        axes[0].set_xlabel('Mean CV Score', fontweight='bold')
        axes[0].set_title(f'Top {top_n} Configurations', fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        all_scores = [r['mean_score'] for r in results]
        axes[1].hist(all_scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1].axvline(max(all_scores), color='red', linestyle='--', linewidth=2, label='Best')
        axes[1].set_xlabel('Mean CV Score', fontweight='bold')
        axes[1].set_ylabel('Count', fontweight='bold')
        axes[1].set_title('Score Distribution', fontweight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'search_results.png', dpi=300, bbox_inches='tight')
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_title(f'{model_name} Confusion Matrix', fontweight='bold')
        fig.savefig(self.output_dir / f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "Model") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        ax.plot(fpr, tpr, linewidth=2.5, color='darkblue', label=f'{model_name} (AUC={auc_score:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.savefig(self.output_dir / f'roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
        return fig
    
    def plot_feature_importance(self, importances: np.ndarray, feature_names: List[str],
                                top_n: int = 20, model_name: str = "Model") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8))
        indices = np.argsort(importances)[-top_n:]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        ax.barh(range(top_n), importances[indices], color=colors, alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(f'{model_name} Top {top_n} Feature Importances', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        fig.savefig(self.output_dir / f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        return fig


class HyperparameterSearcher:
    def __init__(self, config_path: str):
        self.config = ConfigParser(config_path)
        self._setup_logging()
        self._setup_output_dir()
        self.visualizer = Visualizer(self.config.get('visualization', {}), self.output_dir)
        self.logger.info(f"Initialized with config: {config_path}")
    
    def _setup_logging(self):
        log_cfg = self.config.get('logging', {})
        self.logger = logging.getLogger('hp_search')
        self.logger.setLevel(getattr(logging, log_cfg.get('level', 'INFO')))
        self.logger.handlers = []
        
        formatter = logging.Formatter(log_cfg.get('format', '%(asctime)s - %(levelname)s - %(message)s'))
        
        if log_cfg.get('console', True):
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        if log_cfg.get('file'):
            fh = logging.FileHandler(log_cfg['file'])
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def _setup_output_dir(self):
        self.output_dir = Path(self.config.get('global', 'output_dir', default='hp_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict:
        self.logger.info("Starting hyperparameter search pipeline")
        results = {}
        
        X_train, X_test, y_train, y_test = self._prepare_data(X, y)
        search_result = self._run_search(X_train, y_train)
        results['search'] = search_result
        
        if self.config.get('learning_curve', 'enabled', default=False):
            lc_result = self._compute_learning_curves(search_result.best_estimator, X_train, y_train)
            results['learning_curve'] = lc_result
        
        test_results = self._evaluate_on_test(search_result.best_estimator, X_test, y_test, feature_names)
        results['test_evaluation'] = test_results
        
        if self.config.get('visualization', 'enabled', default=True):
            self._generate_visualizations(results, feature_names)
        
        self._save_results(results)
        self.logger.info("Pipeline complete")
        return results
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_cfg = self.config.get('data', {})
        random_seed = self.config.get('global', 'random_seed')
        
        self.logger.info(f"Preparing data: {X.shape[0]} samples, {X.shape[1]} features")
        
        if data_cfg.get('handle_missing') == 'mean':
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
        elif data_cfg.get('handle_missing') == 'zero':
            X = np.nan_to_num(X, 0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=data_cfg.get('test_size', 0.2),
            random_state=random_seed,
            stratify=y if data_cfg.get('stratify', True) else None
        )
        
        self.logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def _run_search(self, X_train: np.ndarray, y_train: np.ndarray) -> SearchResult:
        search_cfg = self.config.get('search', {})
        model_cfg = self.config.get('model', {})
        cv_cfg = self.config.get('cross_validation', 'search_cv', default={})
        global_cfg = self.config.get('global', {})
        random_seed = global_cfg.get('random_seed')
        
        model_type = model_cfg.get('type', 'xgboost')
        strategy = search_cfg.get('strategy', 'random')
        
        self.logger.info(f"Running {strategy} search for {model_type}")
        
        param_space = self.config.get('param_space', model_type, default={})
        fixed_params = model_cfg.get('fixed_params', {})
        
        cv = StratifiedKFold(
            n_splits=cv_cfg.get('n_splits', 3),
            shuffle=cv_cfg.get('shuffle', True),
            random_state=random_seed
        )
        
        subsample_ratio = self.config.get('data', 'search_subsample_ratio')
        if subsample_ratio and subsample_ratio < 1.0:
            n_subsample = int(len(X_train) * subsample_ratio)
            rng = np.random.RandomState(random_seed) if random_seed else np.random
            indices = rng.choice(len(X_train), n_subsample, replace=False)
            X_search, y_search = X_train[indices], y_train[indices]
            self.logger.info(f"Using {n_subsample} samples for search ({subsample_ratio*100:.0f}%)")
        else:
            X_search, y_search = X_train, y_train
        
        checkpoint_file = self.output_dir / 'search_checkpoint.pkl'
        def save_checkpoint(results):
            if self.config.get('global', 'checkpoint_enabled', default=True):
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(results, f)
        
        start_time = time.time()
        n_iter = search_cfg.get('n_iter', 30)
        
        if strategy == 'sklearn_random':
            sklearn_searcher = SklearnRandomizedSearch(
                model_name=model_type,
                param_space=param_space,
                fixed_params=fixed_params,
                cv=cv,
                scoring=search_cfg.get('scoring', 'f1'),
                n_jobs=global_cfg.get('n_jobs', -1),
                verbose=global_cfg.get('verbose', 1),
                random_state=random_seed
            )
            all_results = sklearn_searcher.search(X_search, y_search, n_iter=n_iter)
        else:
            searcher = CrossValidationSearch(
                model_name=model_type,
                param_space=param_space,
                fixed_params=fixed_params,
                cv=cv,
                scoring=search_cfg.get('scoring', 'f1'),
                n_jobs=global_cfg.get('n_jobs', -1),
                verbose=global_cfg.get('verbose', 1),
                random_state=random_seed
            )
            
            if strategy == 'grid':
                all_results = searcher.search_grid(X_search, y_search, checkpoint_callback=save_checkpoint)
            else:
                all_results = searcher.search_random(X_search, y_search, n_iter=n_iter, checkpoint_callback=save_checkpoint)
        
        elapsed = time.time() - start_time
        best_result = max(all_results, key=lambda x: x['mean_score'])
        
        best_params = {**fixed_params, **best_result['params']}
        if random_seed is not None:
            best_params['random_state'] = random_seed
        best_model = ModelRegistry.create(model_type, best_params)
        best_model.fit(X_train, y_train)
        
        self.logger.info(f"Search complete in {elapsed/60:.2f} minutes")
        self.logger.info(f"Best score: {best_result['mean_score']:.4f}")
        self.logger.info(f"Best params: {best_result['params']}")
        
        return SearchResult(
            best_params=best_result['params'],
            best_score=best_result['mean_score'],
            best_score_std=best_result['std_score'],
            all_results=all_results,
            best_estimator=best_model,
            search_time_seconds=elapsed,
            n_configurations_tested=len(all_results)
        )
    
    def _compute_learning_curves(self, estimator: Any, X_train: np.ndarray, y_train: np.ndarray) -> LearningCurveResult:
        self.logger.info("Computing learning curves...")
        random_seed = self.config.get('global', 'random_seed')
        analyzer = LearningCurveAnalyzer(self.config.get('learning_curve', {}), self.config.get('plateau', {}))
        result = analyzer.compute(estimator, X_train, y_train, random_state=random_seed)
        
        if result.plateau_detected:
            self.logger.warning(f"Plateau detected at index {result.plateau_index}")
        return result
    
    def _evaluate_on_test(self, estimator: Any, X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: Optional[List[str]]) -> Dict:
        self.logger.info("Evaluating on test set...")
        
        y_pred = estimator.predict(X_test)
        y_proba = estimator.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        self.logger.info(f"Test F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        return {
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'classification_report': classification_report(y_test, y_pred, target_names=['Human', 'AI'], output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def _generate_visualizations(self, results: Dict, feature_names: Optional[List[str]]):
        self.logger.info("Generating visualizations...")
        model_name = self.config.get('model', 'type', default='Model')
        
        if self.config.get('visualization', 'plots', 'search_history', default=True):
            self.visualizer.plot_search_results(results['search'].all_results)
        
        if 'learning_curve' in results:
            self.visualizer.plot_learning_curves(results['learning_curve'], model_name)
        
        if self.config.get('visualization', 'plots', 'confusion_matrices', default=True):
            self.visualizer.plot_confusion_matrix(
                results['test_evaluation']['y_true'],
                results['test_evaluation']['y_pred'],
                model_name
            )
        
        if self.config.get('visualization', 'plots', 'roc_curves', default=True):
            self.visualizer.plot_roc_curve(
                results['test_evaluation']['y_true'],
                results['test_evaluation']['y_proba'],
                model_name
            )
        
        if (self.config.get('visualization', 'plots', 'feature_importance', default=True)
            and feature_names and hasattr(results['search'].best_estimator, 'feature_importances_')):
            self.visualizer.plot_feature_importance(
                results['search'].best_estimator.feature_importances_,
                feature_names,
                model_name=model_name
            )
    
    def _save_results(self, results: Dict):
        save_cfg = self.config.get('save', {})
        
        if not save_cfg.get('enabled', True):
            self.logger.info("Saving disabled")
            return
        
        if save_cfg.get('cv_results', True):
            pd.DataFrame(results['search'].all_results).to_csv(self.output_dir / 'cv_results.csv', index=False)
        
        if save_cfg.get('best_params', True):
            with open(self.output_dir / 'best_params.yaml', 'w') as f:
                yaml.dump(results['search'].best_params, f)
        
        if save_cfg.get('best_model', True):
            with open(self.output_dir / 'best_model.pkl', 'wb') as f:
                pickle.dump(results['search'].best_estimator, f)
        
        if save_cfg.get('test_metrics', True):
            with open(self.output_dir / 'test_metrics.yaml', 'w') as f:
                yaml.dump(results['test_evaluation']['metrics'], f)
        
        self.logger.info(f"Results saved to {self.output_dir}")

def quick_search(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'xgboost',
    n_iter: int = 20,
    cv_splits: int = 3,
    scoring: str = 'f1',
    random_state: Optional[int] = None,
    use_sklearn: bool = False
) -> SearchResult:
    """Quick hyperparameter search without YAML configuration."""
    default_spaces = {
        'xgboost': {
            'n_estimators': {'type': 'discrete', 'values': [100, 300, 500, 800]},
            'max_depth': {'type': 'int_range', 'min': 3, 'max': 12},
            'learning_rate': {'type': 'log_uniform', 'min': 0.01, 'max': 0.3},
            'subsample': {'type': 'uniform', 'min': 0.6, 'max': 1.0},
            'colsample_bytree': {'type': 'uniform', 'min': 0.6, 'max': 1.0},
        },
        'random_forest': {
            'n_estimators': {'type': 'discrete', 'values': [100, 300, 500]},
            'max_depth': {'type': 'discrete', 'values': [6, 8, 10, 11, 12, 13, 14, 15, 20, 30, None]},
            'min_samples_split': {'type': 'discrete', 'values': [2, 5, 10]},
            'min_samples_leaf': {'type': 'discrete', 'values': [1, 2, 4]},
        },
    }
    
    if model_type not in default_spaces:
        raise ValueError(f"No default space for {model_type}")
    
    _, default_fixed = ModelRegistry.get(model_type)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    start = time.time()
    
    if use_sklearn:
        sklearn_searcher = SklearnRandomizedSearch(
            model_name=model_type,
            param_space=default_spaces[model_type],
            fixed_params=default_fixed,
            cv=cv,
            scoring=scoring,
            verbose=1,
            random_state=random_state
        )
        all_results = sklearn_searcher.search(X, y, n_iter=n_iter)
    else:
        searcher = CrossValidationSearch(
            model_name=model_type,
            param_space=default_spaces[model_type],
            fixed_params=default_fixed,
            cv=cv,
            scoring=scoring,
            verbose=1,
            random_state=random_state
        )
        all_results = searcher.search_random(X, y, n_iter=n_iter)
    
    elapsed = time.time() - start
    best_result = max(all_results, key=lambda x: x['mean_score'])
    
    best_params = {**default_fixed, **best_result['params']}
    if random_state is not None:
        best_params['random_state'] = random_state
    best_model = ModelRegistry.create(model_type, best_params)
    best_model.fit(X, y)
    
    return SearchResult(
        best_params=best_result['params'],
        best_score=best_result['mean_score'],
        best_score_std=best_result['std_score'],
        all_results=all_results,
        best_estimator=best_model,
        search_time_seconds=elapsed,
        n_configurations_tested=len(all_results)
    )


if __name__ == "__main__":
    print("Hyperparameter Search Utilities")
    print("-" * 40)
    print("\nAvailable models:", ModelRegistry.list_available())
    print("\nSearch strategies: 'random', 'grid', 'sklearn_random'")
    print("\nUsage:")
    print("  searcher = HyperparameterSearcher('config.yml')")
    print("  results = searcher.run(X, y, feature_names)")
    print()
    print("  result = quick_search(X, y, random_state=42)")
    print("  result = quick_search(X, y, use_sklearn=True, random_state=42)")