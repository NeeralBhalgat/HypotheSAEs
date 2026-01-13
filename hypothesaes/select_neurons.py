"""Methods for selecting relevant neurons based on target variables."""

import time
import numpy as np
from typing import List, Optional, Callable, Tuple
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

def select_neurons_lasso(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    classification: bool = False,
    alpha: Optional[float] = None,
    max_iter: int = 1000,
    verbose: bool = False
) -> Tuple[List[int], List[float]]:
    """
    Select neurons using an L1-regularized linear model (LASSO), which produces sparse coefficient vectors.
    Returns (indices, coefficients) tuple.
    
    Args:
        activations: Neuron activation matrix (n_samples, n_neurons)
        target: Target variable (n_samples,)
        n_select: Number of neurons to select
        classification: Whether this is a classification task
        alpha: LASSO alpha (if None, searches for alpha yielding n_select features)
        max_iter: Maximum iterations for LASSO
        verbose: Whether to print progress of alpha search
    
    Returns:
        Indices of selected neurons and corresponding coefficients
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)
    
    if alpha is not None:
        if classification:
            model = LogisticRegression(
                penalty='l1', 
                solver='liblinear',
                C=1/alpha,
                max_iter=max_iter
            )
        else:
            model = Lasso(alpha=alpha, max_iter=max_iter)
        
        model.fit(X_scaled, target)
        coef = model.coef_.flatten()
        
    else:
        alpha_low, alpha_high = 1e-6, 1e4
        
        if verbose:
            print(f"{'LASSO iteration':>8} {'L1 Alpha':>10} {'# Features':>10} {'Time (s)':>10}")
            print("-" * 40)
        
        for iteration in range(20):
            iter_start_time = time.time()
            alpha = np.sqrt(alpha_low * alpha_high)
            
            if classification:
                model = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=1/alpha,
                    max_iter=max_iter
                )
            else:
                model = Lasso(alpha=alpha, max_iter=max_iter)
            
            model.fit(X_scaled, target)
            coef = model.coef_.flatten()
            n_nonzero = np.sum(coef != 0)
            iter_time = time.time() - iter_start_time
            
            if verbose:
                print(f"{iteration:8d} {alpha:10.2e} {n_nonzero:10d} {iter_time:10.2f}")
            
            if n_nonzero == n_select:
                break
            elif n_nonzero < n_select:
                alpha_high = alpha
            else:
                alpha_low = alpha
        
        if n_nonzero != n_select and verbose:
            print(f"Warning: Search ended with {n_nonzero} features (target: {n_select})")
    
    sorted_indices = np.argsort(-np.abs(coef))[:n_select]
    selected_coefs = coef[sorted_indices]
    
    return sorted_indices.tolist(), selected_coefs.tolist()

def select_neurons_correlation(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """Select neurons with highest correlation with target."""
    correlations = np.array([
        pearsonr(activations[:, i], target)[0]
        for i in range(activations.shape[1])
    ])
    
    sorted_indices = np.argsort(-np.abs(correlations))[:n_select]
    selected_correlations = correlations[sorted_indices]
    
    return sorted_indices.tolist(), selected_correlations.tolist()

def select_neurons_separation_score(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    n_top_activating: int = 100,
    n_zero_activating: Optional[int] = None,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """Select neurons based on separation between top activations and zero activations."""
    scores = []
    for i in range(activations.shape[1]):
        neuron_acts = activations[:, i]
        sorted_indices = np.argsort(-neuron_acts)
        
        top_mean = np.mean(target[sorted_indices[:n_top_activating]])
        
        zero_indices = neuron_acts == 0
        if n_zero_activating is not None:
            zero_idx = np.where(zero_indices)[0]
            rand_zero_idx = np.random.choice(zero_idx, size=n_zero_activating, replace=False)
            zero_mean = np.mean(target[rand_zero_idx])
        else:
            zero_mean = np.mean(target[zero_indices])
            
        scores.append(top_mean - zero_mean)
    
    scores = np.array(scores)
    sorted_indices = np.argsort(-np.abs(scores))[:n_select]
    selected_scores = scores[sorted_indices]
    
    return sorted_indices.tolist(), selected_scores.tolist()

def select_neurons_custom(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    metric_fn: Callable[[np.ndarray, np.ndarray], float]
) -> Tuple[List[int], List[float]]:
    """Select neurons using a custom metric function."""
    scores = np.array([
        metric_fn(activations[:, i], target)
        for i in range(activations.shape[1])
    ])
    
    sorted_indices = np.argsort(scores)[-n_select:]
    selected_scores = scores[sorted_indices]
    
    return sorted_indices.tolist(), selected_scores.tolist()

def select_neurons(
    activations: np.ndarray,
    target: np.ndarray,
    n_select: int,
    method: str = "lasso",
    classification: bool = False,
    **kwargs
) -> Tuple[List[int], List[float]]:
    """Select neurons using specified method."""
    if classification and len(np.unique(target)) > 2:
        raise ValueError("classification=True, but the target variable has more than 2 classes. We currently do not support multi-class classification, but you can convert to a one-vs-rest binary classification.")

    if method == "lasso":
        return select_neurons_lasso(
            activations=activations,
            target=target,
            n_select=n_select,
            classification=classification,
            **kwargs
        )
    elif method == "correlation":
        return select_neurons_correlation(
            activations=activations,
            target=target,
            n_select=n_select,
            **kwargs
        )
    elif method == "separation_score":
        return select_neurons_separation_score(
            activations=activations,
            target=target,
            n_select=n_select,
            **kwargs
        )
    elif method == "custom":
        if "metric_fn" not in kwargs:
            raise ValueError("Must provide metric_fn for custom method")
        return select_neurons_custom(
            activations=activations,
            target=target,
            n_select=n_select,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")