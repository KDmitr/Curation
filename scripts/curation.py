import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score, roc_auc_score,
                           r2_score, mean_squared_error, mean_absolute_error)
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import clone
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from typing import Callable, Any, Tuple, Optional, List, Dict, Union
import warnings
from synthcity.metrics.representations.OneClass import OneClassLayer
warnings.filterwarnings('ignore')
import torch
from synthcity.utils.constants import DEVICE
from torch.utils.data import DataLoader
from synthcity.plugins.core.dataloader import GenericDataLoader
from xgboost import XGBClassifier, XGBRegressor


class MultiObjectiveSubsetOptimizer:
    """
    A class for multi-criteria search for the optimal subsample of training data
     that optimizes several metrics simultaneously with support for Leave-One-Out selection
    """
    
    def __init__(self, 
                 objectives: Union[List[Dict], Dict] = None,
                 aggregation_method: str = 'weighted_sum',
                 weights: List[float] = None,
                 pareto_front: bool = False,
                 min_subset_size: int = 5,
                 max_iterations: int = None,
                 random_search: bool = False,
                 random_state: int = 42,
                 debug_mode: bool = False,
                 use_loo_selection: bool = False,
                 loo_strategy: str = 'backward',
                 loo_stopping_criterion: str = 'no_improvement',
                 loo_patience: int = 3,
                 task_type: str = 'auto',  
                 ml_weight: float = 1.0,   
                 privacy_weight: float = 1.0,
                 fixed_size: bool = True):  
        """
        Args:
            objectives: A list or dictionary of target functions
            aggregation_method: Aggregation method ('weighted_sum', 'product', 'min', 'max', 'pareto')
            weights: Weights for objective functions (used with weighted_sum)
            pareto_front: If True, it returns a set of Pareto-optimal solutions.
            min_subset_size: Minimum subsample size
            max_iterations: Maximum number of iterations for a random search
            random_search: Use a random search instead of a complete search
            random_state: 
            debug_mode: True to enable debugging prints
            use_loo_selection: Use Leave-One-Out to select a subsample
            loo_strategy: The LOO Strategy ('backward', 'forward', 'bidirectional')
            loo_stopping_criterion: Criteria for stopping ('no_improvement', 'min_size', 'max_iterations')
            loo_patience: Number of iterations without improvement to stop
            task_type: Task type ('classification', 'regression', 'auto')
            ml_weight: Weight for ML metrics in Pareto optimization
            privacy_weight: Weight for privacy metrics in Pareto optimization
        """
        self.task_type = task_type
        self.ml_weight = ml_weight
        self.privacy_weight = privacy_weight
        
        
        if objectives is None and pareto_front:
            self.objectives = self._create_default_pareto_objectives()
            self.weights = [ml_weight, privacy_weight]
        else:
            self.objectives = self._normalize_objectives(objectives)
            self.weights = weights or [1.0] * len(self.objectives)
            
        self.aggregation_method = aggregation_method
        self.pareto_front = pareto_front
        self.min_subset_size = min_subset_size
        self.max_iterations = max_iterations
        self.random_search = random_search
        self.random_state = random_state
        self.debug_mode = debug_mode
        self.use_loo_selection = use_loo_selection
        self.loo_strategy = loo_strategy
        self.loo_stopping_criterion = loo_stopping_criterion
        self.loo_patience = loo_patience
        self.fixed_size=fixed_size
        

        self.pareto_solutions = []
        self.pareto_scores = []
        
        
        self.loo_history = []
        
        np.random.seed(random_state)
        
        
        self._validate_parameters()
    
    def _create_default_pareto_objectives(self):
        """Creates standard objective functions for Pareto optimization"""
        if self.task_type == 'classification':
            return [
                {
                    'func': roc_common,
                    'maximize': True,
                    'weight': self.ml_weight,
                    'name': 'ROC_AUC_Score',
                    'model': None,
                    'metric': None,
                    'multiclass_average': 'ovr',
                    'task_type': 'classification'
                },
                {
                    'func': identifiability,
                    'maximize': True,  
                    'weight': self.privacy_weight,
                    'name': 'Privacy_Score',
                    'model': None,
                    'metric': None,
                    'multiclass_average': 'ovr',
                    'task_type': 'classification'
                }
            ]
        elif self.task_type == 'regression':
            return [
                {
                    'func': r2_common,
                    'maximize': True,
                    'weight': self.ml_weight,
                    'name': 'R2_Score',
                    'model': None,
                    'metric': None,
                    'multiclass_average': 'ovr',
                    'task_type': 'regression'
                },
                {
                    'func': identifiability,
                    'maximize': True, 
                    'weight': self.privacy_weight,
                    'name': 'Privacy_Score',
                    'model': None,
                    'metric': None,
                    'multiclass_average': 'ovr',
                    'task_type': 'regression'
                }
            ]
        else:
            raise ValueError("For Pareto optimization, task_type must be specified by default: 'classification' or 'regression'")
        
    def _normalize_objectives(self, objectives):
        
        if objectives is None:
            
            return [{
                'func': self._default_f1_objective,
                'maximize': True,
                'weight': 1.0,
                'name': 'f1_score',
                'model': None,
                'metric': None,
                'multiclass_average': 'ovr',
                'task_type': 'auto'
            }]
        
        if isinstance(objectives, dict):
            objectives = [objectives]
            
        normalized = []
        for i, obj in enumerate(objectives):
            if callable(obj):
              
                normalized.append({
                    'func': obj,
                    'maximize': True,
                    'weight': 1.0,
                    'name': f'objective_{i}',
                    'model': None,
                    'metric': None,
                    'multiclass_average': 'ovr',
                    'task_type': 'auto'
                })
            elif isinstance(obj, dict):
                
                normalized.append({
                    'func': obj.get('func'),
                    'maximize': obj.get('maximize', True),
                    'weight': obj.get('weight', 1.0),
                    'name': obj.get('name', f'objective_{i}'),
                    'model': obj.get('model', None),
                    'metric': obj.get('metric', None),
                    'multiclass_average': obj.get('multiclass_average', 'ovr'),
                    'task_type': obj.get('task_type', 'auto')
                })
            else:
                raise ValueError(f"Unsupported type of target function: {type(obj)}")
                
        return normalized
    
    def _default_f1_objective(self, X_subset, y_subset, X_test, y_test, target):
        """The default objective function is F1 score with logistic regression"""
        try:
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
            model.fit(X_subset, y_subset)
            y_pred = model.predict(X_test)
            return f1_score(y_test, y_pred, average='weighted')
        except:
            return 0.0
    
    def _create_ml_objective(self, model, metric_func, multiclass_average='ovr', task_type='auto'):
        """
        Creates a target function for an ML model with classification and regression support
        
        Args:
            model: The sklearn or XGBoost model
            metric_func: The metric function
            multiclass_average: Strategy for multiclass classification of ROC-AUC ('ovr', 'ovo')
            task_type: Task type ('classification', 'regression', 'auto')
  
        """
        def ml_objective(X_subset, y_subset, X_test, y_test, target):
            try:
                model_clone = clone(model)
                
                if hasattr(model_clone, 'set_params'):
                    if 'XGB' in str(type(model_clone)):
                        model_clone.set_params(verbosity=0)
                
                model_clone.fit(X_subset, y_subset)
               
                if task_type == 'auto':
                    if hasattr(model_clone, 'predict_proba') or 'Classifier' in str(type(model_clone)):
                        detected_task = 'classification'
                    else:
                        detected_task = 'regression'
                else:
                    detected_task = task_type
                
                if detected_task == 'regression':
                    # Регрессия
                    y_pred = model_clone.predict(X_test)
                    
                    if metric_func.__name__ == 'r2_score':
                        return metric_func(y_test, y_pred)
                    elif metric_func.__name__ == 'mean_squared_error':
                        return metric_func(y_test, y_pred)
                    elif metric_func.__name__ == 'mean_absolute_error':
                        return metric_func(y_test, y_pred)
                    else:
                        return metric_func(y_test, y_pred)
                        
                else:
                    # Classification
                    if metric_func.__name__ == 'roc_auc_score':
                        
                        unique_classes = np.unique(y_test)
                        
                        if len(unique_classes) < 2:
                            return 0.5  
                        elif len(unique_classes) == 2:
                            
                            if hasattr(model_clone, 'predict_proba'):
                                y_pred_proba = model_clone.predict_proba(X_test)[:, 1]
                            elif hasattr(model_clone, 'decision_function'):
                                y_pred_proba = model_clone.decision_function(X_test)
                            else:
                                
                                y_pred_proba = model_clone.predict(X_test)
                            return metric_func(y_test, y_pred_proba)
                        else:
                            
                            if hasattr(model_clone, 'predict_proba'):
                                y_pred_proba = model_clone.predict_proba(X_test)
                                return metric_func(y_test, y_pred_proba, 
                                                 multi_class=multiclass_average, average='macro')
                            elif hasattr(model_clone, 'decision_function'):
                                y_pred_scores = model_clone.decision_function(X_test)
                                return metric_func(y_test, y_pred_scores, 
                                                 multi_class=multiclass_average, average='macro')
                            else:
                                
                                return 0.5
                    else:
                        
                        y_pred = model_clone.predict(X_test)
                        
                        if metric_func.__name__ in ['f1_score', 'precision_score', 'recall_score']:
                            
                            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
                            if len(unique_classes) > 2:
                                return metric_func(y_test, y_pred, average='macro', zero_division=0)
                            else:
                                return metric_func(y_test, y_pred, average='binary', zero_division=0)
                        else:
                            return metric_func(y_test, y_pred)
                            
            except Exception as e:
                if self.debug_mode:
                    print(f"[DEBUG] Error in ML objective: {e}")
                return 0.0
        
        return ml_objective
    
    def _validate_parameters(self):
        """Validation of input parameters"""
        if len(self.weights) != len(self.objectives):
            raise ValueError(f"Number of weights ({len(self.weights)}) must match the number of objective functions ({len(self.objectives)})")
        
        if self.aggregation_method not in ['weighted_sum', 'product', 'min', 'max', 'pareto']:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
        
        if self.loo_strategy not in ['backward', 'forward', 'bidirectional']:
            raise ValueError(f"Unsupported LOO strategy: {self.loo_strategy}")
        
        if self.loo_stopping_criterion not in ['no_improvement', 'min_size', 'max_iterations']:
            raise ValueError(f"Unsupported LOO stop criterion: {self.loo_stopping_criterion}")
            
        for obj in self.objectives:
            if obj['func'] is None and (obj['model'] is None or obj['metric'] is None):
                raise ValueError("The objective function cannot be None, or the model and metric must be specified.")
    
    def _evaluate_objectives(self, X_subset, y_subset, X_test, y_test, target):
        """Evaluates all objective functions for a given subsample"""
        scores = []
        
        for obj in self.objectives:
            try:
                
                if obj['model'] is not None and obj['metric'] is not None:
                    multiclass_avg = obj.get('multiclass_average', 'ovr')
                    task_type = obj.get('task_type', 'auto')
                    ml_objective = self._create_ml_objective(obj['model'], obj['metric'], 
                                                           multiclass_avg, task_type)
                    raw_score = ml_objective(X_subset, y_subset, X_test, y_test, target)
                else:
                   
                    raw_score = round(obj['func'](X_subset, y_subset, X_test, y_test, target)*obj['weight'],2)
                
                normalized_score = raw_score if obj['maximize'] else -raw_score
                scores.append(normalized_score)
                
                if self.debug_mode:
                    direction = "max" if obj['maximize'] else "min"
                    print(f"[DEBUG] {obj['name']}: {raw_score:.4f} ({direction}) -> {normalized_score:.4f}")
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"[DEBUG] Ошибка в {obj['name']}: {e}")
                scores.append(-np.inf)
        
        return np.array(scores)
    
    def _aggregate_scores(self, scores):
        """Aggregates multiple scores into a single composite score"""
        if self.aggregation_method == 'weighted_sum':
            return np.sum(np.array(scores) * np.array(self.weights))
        
        elif self.aggregation_method == 'product':
           
            shifted_scores = scores - np.min(scores) + 1e-6
            weighted_scores = np.power(shifted_scores, self.weights)
            return np.prod(weighted_scores)
        
        elif self.aggregation_method == 'min':
         
            return np.min(np.array(scores) * np.array(self.weights))
        
        elif self.aggregation_method == 'max':
           
            return np.max(np.array(scores) * np.array(self.weights))
        
        elif self.aggregation_method == 'pareto':
            
            return scores
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _is_pareto_optimal(self, scores, pareto_scores_list):
        """Checks whether the solution is Pareto-optimal"""
        for existing_scores in pareto_scores_list:
            
            if np.all(existing_scores >= scores) and np.any(existing_scores > scores):
                return False
        return True
    
    def _update_pareto_front(self, scores, subset):
        """Updates the Pareto front with a new solution"""
        non_dominated_indices = []
        for i, existing_scores in enumerate(self.pareto_scores):
            
            if not (np.all(scores >= existing_scores) and np.any(scores > existing_scores)):
                non_dominated_indices.append(i)
        
        self.pareto_scores = [self.pareto_scores[i] for i in non_dominated_indices]
        self.pareto_solutions = [self.pareto_solutions[i] for i in non_dominated_indices]
    
        self.pareto_scores.append(scores.copy())
        self.pareto_solutions.append(subset.copy())
    
    def _loo_backward_selection(self, X_train, y_train, X_test, y_test, target):
        """
        Reverse LOO selection: we start with a complete sample and iteratively remove the worst samples
        """
        print(f"We start the reverse LOO selection with  {len(X_train)} samples")
        
        current_indices = list(X_train.index)
        current_subset = X_train.copy()
        current_y_subset = y_train.copy()
        
        current_scores = self._evaluate_objectives(current_subset, current_y_subset, X_test, y_test, target)
        current_composite_score = self._aggregate_scores(current_scores)
        
        best_composite_score = current_composite_score
        best_subset = current_subset.copy()
        best_scores = current_scores.copy()
        
        self.loo_history = [{
            'iteration': 0,
            'action': 'initial',
            'subset_size': len(current_subset),
            'composite_score': current_composite_score,
            'individual_scores': current_scores.copy(),
            'removed_index': None
        }]
        
        no_improvement_count = 0
        iteration = 1
        
        while len(current_indices) > self.min_subset_size:
            if self.debug_mode:
                print(f"[DEBUG] LOO iteration {iteration}, subsample size: {len(current_indices)}")
            
            best_removal_score = -np.inf
            best_removal_index = None
            best_removal_scores = None
            
            for idx_to_remove in current_indices:
                
                temp_indices = [idx for idx in current_indices if idx != idx_to_remove]
                temp_subset = X_train.loc[temp_indices]
                temp_y_subset = y_train.loc[temp_indices]
                
                temp_scores = self._evaluate_objectives(temp_subset, temp_y_subset, X_test, y_test, target)
                temp_composite_score = self._aggregate_scores(temp_scores)
                
                if self.debug_mode:
                    print(f"[DEBUG] Sample Removal {idx_to_remove}: score = {temp_composite_score:.4f}")
                
                if temp_composite_score > best_removal_score:
                    best_removal_score = temp_composite_score
                    best_removal_index = idx_to_remove
                    best_removal_scores = temp_scores.copy()
            
            if best_removal_score > current_composite_score:
                
                current_indices.remove(best_removal_index)
                current_subset = X_train.loc[current_indices]
                current_y_subset = y_train.loc[current_indices]
                current_composite_score = best_removal_score
                current_scores = best_removal_scores
                
               
                if best_removal_score > best_composite_score:
                    best_composite_score = best_removal_score
                    best_subset = pd.concat([current_subset.copy(),current_y_subset.copy()],axis=1)
                    best_scores = current_scores.copy()
                
                no_improvement_count = 0
                
                print(f"Iteration {iteration}: The sample was deleted {best_removal_index}, "
                      f"subsample size: {len(current_indices)}, score: {best_removal_score:.4f}")
                
            else:
                no_improvement_count += 1
                print(f"Iteration {iteration}: there is no improvement when removing any sample")
                
                if self.loo_stopping_criterion == 'no_improvement' and no_improvement_count >= self.loo_patience:
                    print(f"Shutdown: no improvement during {self.loo_patience} iterations")
                    break
            
            self.loo_history.append({
                'iteration': iteration,
                'action': 'add',
                'subset_size': len(current_indices),
                'composite_score': current_composite_score,
                'individual_scores': current_scores.copy(),
                'added_index': best_addition_index
            })
            
            iteration += 1
            
            if self.loo_stopping_criterion == 'max_iterations' and iteration > (self.max_iterations or 100):
                print(f"Stop: the maximum number of iterations has been reached")
                break
        
        return best_composite_score, best_subset, best_scores
    
    def _loo_bidirectional_selection(self, X_train, y_train, X_test, y_test, target):
        """
        Bidirectional LOO selection: combines direct and reverse selection
        """
        print(f"We start bidirectional LOO sampling from {len(X_train)} samples")
        
        print("Phase 1: Direct selection")
        forward_score, forward_subset, forward_scores = self._loo_forward_selection(X_train, y_train, X_test, y_test, target)
        forward_history = self.loo_history.copy()
        
        print("Phase 2: Reverse selection from the result of direct selection")
        
        temp_X_train = forward_subset.copy()
        temp_y_train = y_train.loc[forward_subset.index]
        
        backward_score, backward_subset, backward_scores = self._loo_backward_selection(temp_X_train, temp_y_train, X_test, y_test, target)
        backward_history = self.loo_history.copy()
        
        self.loo_history = forward_history + [{'phase': 'backward_start'}] + backward_history
        
        if backward_score > forward_score:
            print(f"The best result after the reverse selection: {backward_score:.4f}")
            return backward_score, backward_subset, backward_scores
        else:
            print(f"The best result after direct selection: {forward_score:.4f}")
            return forward_score, forward_subset, forward_scores
    
    def _loo_selection_optimize(self, X_train, y_train, X_test, y_test, target):
        """Performs LOO subsampling selection according to the strategy"""
        if self.loo_strategy == 'backward':
            return self._loo_backward_selection(X_train, y_train, X_test, y_test, target)
        elif self.loo_strategy == 'forward':
            return self._loo_forward_selection(X_train, y_train, X_test, y_test, target)
        elif self.loo_strategy == 'bidirectional':
            return self._loo_bidirectional_selection(X_train, y_train, X_test, y_test, target)
        else:
            raise ValueError(f"Unknown LOO Strategy: {self.loo_strategy}")
    
    def _random_search(self, X_train, y_train, X_test, y_test, target):
        """Random search with multi-criteria optimization support"""
        n_samples = len(X_train)
        max_iter = self.max_iterations if self.max_iterations else min(10000, 2**(n_samples//2))
        
        best_composite_score = -np.inf
        best_subset = None
        best_scores = None
        
        if self.pareto_front:
            self.pareto_solutions = []
            self.pareto_scores = []
        
        print(f"Starting a multi-criteria random search with {max_iter} iterations")
        print(f"Number of target functions: {len(self.objectives)}")
        print(f"Aggregation method: {self.aggregation_method}")
        
        for iteration in range(max_iter):
            if iteration % 1000 == 0:
                if self.pareto_front:
                    print(f"Iteration {iteration}/{max_iter}, the size of the Pareto front: {len(self.pareto_solutions)}")
                else:
                    print(f"Iteration {iteration}/{max_iter}, best composite score: {best_composite_score:.4f}")
            
            # Randomly selecting the size and indexes of the subsample
            if self.fixed_size:
            #subset_size = np.random.randint(self.min_subset_size, n_samples + 1)
                subset_indices = np.random.choice(n_samples, size=self.min_subset_size, replace=False)
            else:
                subset_size = np.random.randint(self.min_subset_size, n_samples + 1)
                subset_indices = np.random.choice(n_samples, size=subset_size, replace=False)

            X_subset = X_train.iloc[subset_indices]
            y_subset = y_train.iloc[subset_indices]

            # Evaluating all objective functions
            scores = self._evaluate_objectives(X_subset, y_subset, X_test, y_test, target)
            
            if self.pareto_front:
                # Pareto optimization
                if self._is_pareto_optimal(scores, self.pareto_scores):
                    self._update_pareto_front(scores, pd.concat([X_subset.copy(),y_subset.copy()],axis=1))
            else:
            
                composite_score = self._aggregate_scores(scores)
                
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_subset = pd.concat([X_subset.copy(),y_subset.copy()],axis=1)
                    best_scores = scores.copy()
        
        if self.pareto_front:
            return None, None 
        else:
            return best_composite_score, best_subset, best_scores
    
    def _exhaustive_search(self, X_train, y_train, X_test, y_test, target):
        """A complete search with support for multi-criteria optimization"""
        n_samples = len(X_train)
        best_composite_score = -np.inf
        best_subset = None
        best_scores = None
        
        if self.pareto_front:
            self.pareto_solutions = []
            self.pareto_scores = []
        
        print(f"Starting a multi-criteria full search for {n_samples} samples")
        
        for subset_size in range(self.min_subset_size, n_samples + 1):
            print(f"Checking the size subsamples {subset_size}")
            
            total_combinations = np.math.comb(n_samples, subset_size)
            if total_combinations > 10000:
                print(f"There are too many combinations ({total_combinations}), switching to random search")
                return self._random_search(X_train, y_train, X_test, y_test, target)
            
            for indices in combinations(range(n_samples), subset_size):
                subset_indices = list(indices)
                X_subset = X_train.iloc[subset_indices]
                y_subset = y_train.iloc[subset_indices]
                
                scores = self._evaluate_objectives(X_subset, y_subset, X_test, y_test, target)
                
                if self.pareto_front:
                    if self._is_pareto_optimal(scores, self.pareto_scores):
                        self._update_pareto_front(scores, pd.concat([X_subset.copy(),y_subset.copy()],axis=1))
                else:
                    composite_score = self._aggregate_scores(scores)
                    
                    if composite_score > best_composite_score:
                        best_composite_score = composite_score
                        best_subset = pd.concat([X_subset.copy(),y_subset.copy()],axis=1)
                        best_scores = scores.copy()
        
        if self.pareto_front:
            return None, None
        else:
            return best_composite_score, best_subset, best_scores
    
    def optimize(self, X_train, y_train, X_test, y_test, target):
        """
        The main function of multi-criteria optimization
        
        Returns:
            If pareto_front=False: Tuple[float, pd.DataFrame, np.array] - composite_score, best subsample, individual_scores
            If pareto_front=True: Tuple[List[pd.DataFrame], List[np.array]] - Pareto-optimal subsamples and their scores
        """
        if len(X_train) < self.min_subset_size:
            raise ValueError(f"The size of the training sample ({len(X_train)}) less than the minimum size of the subsample ({self.min_subset_size})")
        
        if self.use_loo_selection:
            if self.pareto_front:
                print("Attention: LOO selection with Pareto optimization is not supported. Aggregated optimization is used.")
                self.pareto_front = False
            
            composite_score, best_subset, individual_scores = self._loo_selection_optimize(X_train, y_train, X_test, y_test, target)
            
            display_scores = []
            for i, (score, obj) in enumerate(zip(individual_scores, self.objectives)):
                original_score = score if obj['maximize'] else -score
                display_scores.append(original_score)
            
            return composite_score, best_subset, np.array(display_scores)
        
        if self.random_search or len(X_train) > 20:
            result = self._random_search(X_train, y_train, X_test, y_test, target)
        else:
            result = self._exhaustive_search(X_train, y_train, X_test, y_test, target)
        
        if self.pareto_front:
            print(f"\n Found {len(self.pareto_solutions)} Pareto-optimal solutions")
            return self.pareto_solutions, self.pareto_scores
        else:
            composite_score, best_subset, individual_scores = result
            
            display_scores = []
            for i, (score, obj) in enumerate(zip(individual_scores, self.objectives)):
                original_score = score if obj['maximize'] else -score
                display_scores.append(original_score)
            
            return composite_score, best_subset, np.array(display_scores)
    
    

    def create_pareto_optimizer(self, task_type='auto', ml_weight=1.0, privacy_weight=1.0, 
                               max_iterations=5000, random_search=True, min_subset_size=5):
        """
        A function for creating a Pareto optimizer with preset functions
        
        Args:
            task_type: 'classification' or 'regression'
            ml_weight: Weight for the ML metric
            privacy_weight: Weight for privacy metrics
            max_iterations: Maximum number of iterations
            random_search: Use random search
            min_subset_size: Minimum subsample size
        
        """
        return MultiObjectiveSubsetOptimizer(
            objectives=None,  
            pareto_front=True,
            task_type=task_type,
            ml_weight=ml_weight,
            privacy_weight=privacy_weight,
            max_iterations=max_iterations,
            random_search=random_search,
            min_subset_size=min_subset_size,
            random_state=self.random_state,
            debug_mode=self.debug_mode,
            fixed_size=self.fixed_size
        )



def create_ml_objective(model, metric_name, maximize=True, weight=1.0, name=None, 
                       multiclass_average='ovr', task_type='auto'):
    
    classification_metrics = {
        'f1': f1_score,
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'roc_auc': roc_auc_score
    }
    
   
    regression_metrics = {
        'r2': r2_score,
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
    
    if metric_name in classification_metrics:
        metric_func = classification_metrics[metric_name]
        detected_task = 'classification'
    elif metric_name in regression_metrics:
        metric_func = regression_metrics[metric_name]
        detected_task = 'regression'
        
        if metric_name in ['mse', 'mae', 'rmse'] and maximize is True:
            maximize = False
    else:
        raise ValueError(f"Unsupported metric: {metric_name}. "
                        f"Available classification metrics: {list(classification_metrics.keys())}. "
                        f"Available regression metrics: {list(regression_metrics.keys())}")
    
    final_task_type = task_type if task_type != 'auto' else detected_task
    
    return {
        'func': None,  
        'model': model,
        'metric': metric_func,
        'maximize': maximize,
        'weight': weight,
        'name': name or f'{metric_name}_score',
        'multiclass_average': multiclass_average,
        'task_type': final_task_type
    }

# Functions for evaluating the quality of ML models
def roc_LR(X_subset, y_subset, X_test, y_test):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_subset, y_subset)
    if len(y_test.value_counts().index) > 2:
        y_pred = model.predict_proba(X_test)
    else:
        y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

def roc_XGB(X_subset, y_subset, X_test, y_test):
   
    model = XGBClassifier(random_state=42, verbosity=0)
    model.fit(X_subset, y_subset)
    if len(y_test.value_counts().index) > 2:
        y_pred = model.predict_proba(X_test)
    else:
        y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')

def roc_common(X_subset, y_subset, X_test, y_test, target):

    """Average ROC-AUC score for LogisticRegression and XGBoost"""

    lr_score = roc_LR(X_subset, y_subset, X_test, y_test)
    xgb_score = roc_XGB(X_subset, y_subset, X_test, y_test)
    tt = np.mean([lr_score, xgb_score])
    if hasattr(roc_common, 'debug') and roc_common.debug:
        print(f"ROC-AUC: LR={lr_score:.4f}, XGB={xgb_score:.4f}, Mean={tt:.4f}")
    return round(tt, 3)

def get_oneclass_model(X_gt: np.ndarray) -> OneClassLayer:

        

        model = OneClassLayer(
            input_dim=X_gt.shape[1],
            rep_dim=X_gt.shape[1],
            center=torch.ones(X_gt.shape[1]) * 10,
        )
        model.fit(X_gt)

       

        return model.to(DEVICE)

def oneclass_predict(model: OneClassLayer, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(X).float().to(DEVICE)).cpu().detach().numpy()

def compute_scores(X_gt: DataLoader, X_syn: DataLoader, emb: str = ""
    ) -> Dict:
        """Compare Wasserstein distance between original data and synthetic data.

        Args:
            orig_data: original data
            synth_data: synthetically generated data

        Returns:
            WD_value: Wasserstein distance
        """
        X_gt_ = X_gt.numpy().reshape(len(X_gt), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)

        if emb == "privacy.identifiability_score.OC":
            emb = f"_{emb}"
            oneclass_model = get_oneclass_model(X_gt_)
            X_gt_ = oneclass_predict(oneclass_model, X_gt_)
            X_syn_ = oneclass_predict(oneclass_model, X_syn_)
        else:
            if emb != "":
                raise RuntimeError(f" Invalid emb {emb}")

        # Entropy computation
        def compute_entropy(labels: np.ndarray) -> np.ndarray:
            value, counts = np.unique(np.round(labels), return_counts=True)
            return entropy(counts)

        # Parameters
        no, x_dim = X_gt_.shape

        # Weights
        W = np.zeros(
            [
                x_dim,
            ]
        )

        for i in range(x_dim):
            W[i] = compute_entropy(X_gt_[:, i])

        # Normalization
        X_hat = X_gt_
        X_syn_hat = X_syn_

        eps = 1e-16
        W = np.ones_like(W)

        for i in range(x_dim):
            X_hat[:, i] = X_gt_[:, i] * 1.0 / (W[i] + eps)
            X_syn_hat[:, i] = X_syn_[:, i] * 1.0 / (W[i] + eps)

        # r_i computation
        nbrs = NearestNeighbors(n_neighbors=2).fit(X_hat)
        distance, _ = nbrs.kneighbors(X_hat)

        # hat{r_i} computation
        nbrs_hat = NearestNeighbors(n_neighbors=1).fit(X_syn_hat)
        distance_hat, _ = nbrs_hat.kneighbors(X_hat)

        # See which one is bigger
        R_Diff = distance_hat[:, 0] - distance[:, 1]
        identifiability_value = np.sum(R_Diff < 0) / float(no)

        return {f"privacy.identifiability_score.{emb}": identifiability_value}

def identifiability(X_train_subset, y_train_subset, X_test, y_test, target):
    """
    A function for assessing identifiability (privacy)

    The implementation of the identifiability function and its auxiliary functions is taken from the synthcity framework.

    Reference: Qian, Zhaozhi and Cebere, Bogdan-Constantin and van der Schaar, Mihaelar,
    'Synthcity: facilitating innovative use cases of synthetic data in different data modalities'
    Paper link: https://arxiv.org/abs/2301.07573

    """
    
    ass=pd.DataFrame()
    df_real = pd.concat([X_test, y_test], axis=1)
    df_syn = pd.concat([X_train_subset, y_train_subset], axis=1)

    X_gt= GenericDataLoader(df_real)
    X_syn = GenericDataLoader(df_syn)

    oc_results = compute_scores(X_gt, X_syn, "privacy.identifiability_score.OC")
    
    score = round(oc_results['privacy.identifiability_score._privacy.identifiability_score.OC'], 2)
        
    return -score  

def r2_LR(X_subset, y_subset, X_test, y_test):
    model = LinearRegression()
    model.fit(X_subset, y_subset)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

def r2_XGB(X_subset, y_subset, X_test, y_test):
   
    model = XGBRegressor(n_jobs=2, verbosity=0, max_depth=3, random_state=0)
    model.fit(X_subset, y_subset)
    y_pred = model.predict(X_test)
    return r2_score(y_test, y_pred)

def r2_common(X_subset, y_subset, X_test, y_test, target):

    """Average R2 score for LinearRegression and XGBoost"""

    lr_score = r2_LR(X_subset, y_subset, X_test, y_test)
    xgb_score = r2_XGB(X_subset, y_subset, X_test, y_test)
    tt = np.mean([lr_score, xgb_score])
    if hasattr(r2_common, 'debug') and r2_common.debug:
        print(f"R²: LR={lr_score:.4f}, XGB={xgb_score:.4f}, Mean={tt:.4f}")
    return round(tt, 3)


# Functions for creating Pareto optimizers

def create_pareto_optimizer_classification(ml_weight=1.0, privacy_weight=1.0, 
                                          max_iterations=5000, random_search=True, 
                                          min_subset_size=100, random_state=42, debug_mode=False,fixed_size=True):
    """
    Creates an optimizer for Pareto optimization in classification problems
    
    Args:
        ml_weight: Weight for the ROC-AUC metric
        privacy_weight: Weight for privacy metrics
        max_iterations: Maximum number of iterations
        random_search: Use random search
        min_subset_size: Minimum subsample size
        random_state:
        debug_mode: Debugging mode

    """
    return MultiObjectiveSubsetOptimizer(
        objectives=None,  
        pareto_front=True,
        task_type='classification',
        ml_weight=ml_weight,
        privacy_weight=privacy_weight,
        max_iterations=max_iterations,
        random_search=random_search,
        min_subset_size=min_subset_size,
        random_state=random_state,
        debug_mode=debug_mode,
        fixed_size=fixed_size
    )

def create_pareto_optimizer_regression(ml_weight=1.0, privacy_weight=0.1, 
                                      max_iterations=5000, random_search=True, 
                                      min_subset_size=100, random_state=42, debug_mode=False,fixed_size=True):
    """
    Creates an optimizer for Pareto optimization in regression problems
    
    Args:
        ml_weight: Weight for the R2 metric
        privacy_weight: Weight for privacy metrics
        max_iterations: Maximum number of iterations
        random_search: Use random search
        min_subset_size: Minimum subsample size
        random_state: 
        debug_mode: Debugging mode
    
    """
    return MultiObjectiveSubsetOptimizer(
        objectives=None,  
        pareto_front=True,
        task_type='regression',
        ml_weight=ml_weight,
        privacy_weight=privacy_weight,
        max_iterations=max_iterations,
        random_search=random_search,
        min_subset_size=min_subset_size,
        random_state=random_state,
        debug_mode=debug_mode,
        fixed_size=fixed_size
    )
