"""
Training logic for linear value functions.

Implements weighted logistic regression with L1/L2 regularization,
cross-validation, and grid search over hyperparameters.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

from src.evaluation.data_loader import TrainingDataset


@dataclass
class TrainingResult:
    """
    Results from a single training run.

    Contains the trained model parameters, regularization settings,
    and performance metrics.
    """
    theta: np.ndarray       # Weight vector (n_features,)
    bias: float             # Bias term
    lambda1: float          # L1 penalty used
    lambda2: float          # L2 penalty used
    train_loss: float       # Weighted log-loss on train set
    val_loss: float         # Weighted log-loss on validation set
    means: np.ndarray       # Feature means for standardization
    stds: np.ndarray        # Feature stds for standardization
    cv_fold_losses: Optional[List[float]] = None  # Per-fold validation losses

    def __repr__(self):
        return (
            f"TrainingResult(λ1={self.lambda1:.4f}, λ2={self.lambda2:.4f}, "
            f"train_loss={self.train_loss:.4f}, val_loss={self.val_loss:.4f})"
        )


@dataclass
class GridSearchResults:
    """
    Results from grid search over regularization parameters.

    Contains all models tried and the best model selected.
    """
    results: List[TrainingResult]
    best_model: TrainingResult
    best_by_metric: str         # 'val_loss'
    n_models_tried: int
    cv_folds: int

    def __repr__(self):
        return (
            f"GridSearchResults(n_models={self.n_models_tried}, "
            f"best: λ1={self.best_model.lambda1:.4f}, λ2={self.best_model.lambda2:.4f}, "
            f"val_loss={self.best_model.val_loss:.4f})"
        )

    def get_sorted_results(self, by='val_loss', ascending=True) -> List[TrainingResult]:
        """
        Get results sorted by a metric.

        Args:
            by: Metric to sort by ('val_loss', 'train_loss', 'lambda1', 'lambda2')
            ascending: Sort in ascending order (default: True)

        Returns:
            Sorted list of TrainingResults
        """
        if by == 'val_loss':
            key_func = lambda r: r.val_loss
        elif by == 'train_loss':
            key_func = lambda r: r.train_loss
        elif by == 'lambda1':
            key_func = lambda r: r.lambda1
        elif by == 'lambda2':
            key_func = lambda r: r.lambda2
        else:
            raise ValueError(f"Unknown metric: {by}")

        return sorted(self.results, key=key_func, reverse=not ascending)


def compute_standardization_stats(
    X: np.ndarray,
    epsilon: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for feature standardization.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        epsilon: Small constant to prevent division by zero

    Returns:
        means: shape (n_features,)
        stds: shape (n_features,)
    """
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    return means, stds


def standardize_features(
    X: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Standardize features: φ̃ = (φ - μ) / (σ + ε)

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        means: Feature means, shape (n_features,)
        stds: Feature stds, shape (n_features,)
        epsilon: Small constant for numerical stability

    Returns:
        Standardized features, same shape as X
    """
    return (X - means) / (stds + epsilon)


def compute_weighted_log_loss(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray
) -> float:
    """
    Compute weighted binary cross-entropy loss.

    Args:
        model: Trained LogisticRegression model
        X: Feature matrix
        y: Labels (0 or 1)
        w: Sample weights

    Returns:
        Weighted log loss
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    return log_loss(y, y_pred_proba, sample_weight=w)


def get_sklearn_params(lambda1: float, lambda2: float) -> dict:
    """
    Map our regularization parameters to sklearn's LogisticRegression params.

    sklearn uses C (inverse regularization strength) and penalty type.
    We use λ₁ (L1) and λ₂ (L2) directly.

    Mapping:
        - L1 only (λ₁ > 0, λ₂ = 0): penalty='l1', C=1/λ₁, solver='liblinear'
        - L2 only (λ₁ = 0, λ₂ > 0): penalty='l2', C=1/λ₂, solver='lbfgs'
        - Both (λ₁ > 0, λ₂ > 0): penalty='elasticnet', C=1/(λ₁+λ₂),
          l1_ratio=λ₁/(λ₁+λ₂), solver='saga'
        - Neither (λ₁ = 0, λ₂ = 0): penalty='none', solver='lbfgs'

    Args:
        lambda1: L1 regularization strength
        lambda2: L2 regularization strength

    Returns:
        Dictionary of sklearn LogisticRegression parameters
    """
    if lambda1 > 0 and lambda2 > 0:
        # ElasticNet: both L1 and L2
        return {
            'penalty': 'elasticnet',
            'solver': 'saga',
            'C': 1.0 / (lambda1 + lambda2),
            'l1_ratio': lambda1 / (lambda1 + lambda2)
        }
    elif lambda1 > 0:
        # L1 only
        return {
            'penalty': 'l1',
            'solver': 'liblinear',
            'C': 1.0 / lambda1
        }
    elif lambda2 > 0:
        # L2 only
        return {
            'penalty': 'l2',
            'solver': 'lbfgs',
            'C': 1.0 / lambda2
        }
    else:
        # No regularization
        return {
            'penalty': 'none',
            'solver': 'lbfgs',
            'C': 1.0
        }


def train_linear_value_function(
    dataset: TrainingDataset,
    lambda1_values: List[float],
    lambda2_values: List[float],
    cv_folds: int = 5,
    epsilon: float = 1e-8,
    random_state: int = 42,
    verbose: bool = False,
    normalize: bool = True
) -> GridSearchResults:
    """
    Train linear value function with grid search over regularization.

    Performs cross-validated grid search over L1/L2 regularization
    parameters, using game-level splits to prevent data leakage.

    Args:
        dataset: Training data
        lambda1_values: L1 penalties to try (e.g., [0.0, 0.01, 0.1, 1.0])
        lambda2_values: L2 penalties to try (e.g., [0.0, 0.01, 0.1, 1.0])
        cv_folds: Number of cross-validation folds (game-level)
        epsilon: Epsilon for feature standardization
        random_state: Random seed for reproducibility
        verbose: Print progress information
        normalize: Whether to standardize features (default: True)

    Returns:
        GridSearchResults with all models and best model selected

    Process:
        1. Compute standardization stats on full training data
        2. For each (λ₁, λ₂) combination:
           a. Create GroupKFold splits (game-level)
           b. For each fold:
              - Standardize features using training fold stats
              - Train LogisticRegression with sample_weight
              - Evaluate on validation fold
           c. Average validation loss across folds
        3. Select best model by validation loss
        4. Retrain on full dataset with best hyperparameters

    Example:
        >>> from src.logging.storage import GameStorage
        >>> from src.evaluation.data_loader import load_training_data
        >>> storage = GameStorage('data')
        >>> dataset = load_training_data(storage, limit=1000)
        >>> results = train_linear_value_function(
        ...     dataset,
        ...     lambda1_values=[0.0, 0.01, 0.1],
        ...     lambda2_values=[0.0, 0.01, 0.1],
        ...     cv_folds=5
        ... )
        >>> print(f"Best model: λ1={results.best_model.lambda1}, "
        ...       f"λ2={results.best_model.lambda2}")
    """
    # Convert dataset to arrays
    X, y, w, game_ids = dataset.to_arrays()

    if verbose:
        print(f"\n=== Training Linear Value Function ===")
        print(f"Dataset: {len(dataset)} examples, {len(dataset.game_ids)} games")
        print(f"Features: {X.shape[1]}")
        print(f"Grid search: {len(lambda1_values)} × {len(lambda2_values)} = "
              f"{len(lambda1_values) * len(lambda2_values)} models")
        print(f"Cross-validation: {cv_folds} folds (game-level)\n")

    # Compute standardization stats on full data
    if normalize:
        means, stds = compute_standardization_stats(X, epsilon)
        X_std = standardize_features(X, means, stds, epsilon)

        if verbose:
            print(f"Feature standardization:")
            print(f"  Means: min={means.min():.3f}, max={means.max():.3f}")
            print(f"  Stds:  min={stds.min():.3f}, max={stds.max():.3f}\n")
    else:
        # No normalization - use raw features
        X_std = X
        means = np.zeros(X.shape[1])
        stds = np.ones(X.shape[1])

        if verbose:
            print(f"Feature normalization: DISABLED (using raw features)\n")

    # Grid search
    results = []
    n_total = len(lambda1_values) * len(lambda2_values)
    n_processed = 0

    for lambda1 in lambda1_values:
        for lambda2 in lambda2_values:
            n_processed += 1

            if verbose:
                print(f"[{n_processed}/{n_total}] Training λ1={lambda1:.4f}, λ2={lambda2:.4f}...")

            # Get sklearn parameters
            sklearn_params = get_sklearn_params(lambda1, lambda2)

            # Cross-validation with game-level splits
            gkf = GroupKFold(n_splits=cv_folds)
            fold_val_losses = []

            for fold_num, (train_idx, val_idx) in enumerate(gkf.split(X_std, y, groups=game_ids)):
                X_train, X_val = X_std[train_idx], X_std[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                w_train, w_val = w[train_idx], w[val_idx]

                # Create and train model
                model = LogisticRegression(
                    **sklearn_params,
                    random_state=random_state,
                    max_iter=1000
                )

                model.fit(X_train, y_train, sample_weight=w_train)

                # Evaluate on validation fold
                val_loss = compute_weighted_log_loss(model, X_val, y_val, w_val)
                fold_val_losses.append(val_loss)

            avg_val_loss = np.mean(fold_val_losses)

            # Retrain on full data for this hyperparameter setting
            model = LogisticRegression(
                **sklearn_params,
                random_state=random_state,
                max_iter=1000
            )
            model.fit(X_std, y, sample_weight=w)

            # Compute train loss
            train_loss = compute_weighted_log_loss(model, X_std, y, w)

            if verbose:
                print(f"    Train loss: {train_loss:.4f}, Val loss: {avg_val_loss:.4f}")

            # Store result
            results.append(TrainingResult(
                theta=model.coef_[0].copy(),
                bias=float(model.intercept_[0]),
                lambda1=lambda1,
                lambda2=lambda2,
                train_loss=train_loss,
                val_loss=avg_val_loss,
                means=means.copy(),
                stds=stds.copy(),
                cv_fold_losses=fold_val_losses
            ))

    # Find best model by validation loss
    best_model = min(results, key=lambda r: r.val_loss)

    if verbose:
        print(f"\n=== Grid Search Complete ===")
        print(f"Best model: λ1={best_model.lambda1:.4f}, λ2={best_model.lambda2:.4f}")
        print(f"  Train loss: {best_model.train_loss:.4f}")
        print(f"  Val loss:   {best_model.val_loss:.4f}")

    return GridSearchResults(
        results=results,
        best_model=best_model,
        best_by_metric='val_loss',
        n_models_tried=len(results),
        cv_folds=cv_folds
    )


def train_single_model(
    dataset: TrainingDataset,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    epsilon: float = 1e-8,
    random_state: int = 42
) -> TrainingResult:
    """
    Train a single model without cross-validation.

    Useful for quick training or when hyperparameters are already known.

    Args:
        dataset: Training data
        lambda1: L1 regularization strength
        lambda2: L2 regularization strength
        epsilon: Epsilon for feature standardization
        random_state: Random seed

    Returns:
        TrainingResult with trained model parameters

    Example:
        >>> result = train_single_model(dataset, lambda1=0.1, lambda2=0.01)
        >>> print(f"Train loss: {result.train_loss:.4f}")
    """
    # Convert to arrays
    X, y, w, _ = dataset.to_arrays()

    # Standardize
    means, stds = compute_standardization_stats(X, epsilon)
    X_std = standardize_features(X, means, stds, epsilon)

    # Get sklearn params
    sklearn_params = get_sklearn_params(lambda1, lambda2)

    # Train model
    model = LogisticRegression(
        **sklearn_params,
        random_state=random_state,
        max_iter=1000
    )
    model.fit(X_std, y, sample_weight=w)

    # Compute train loss
    train_loss = compute_weighted_log_loss(model, X_std, y, w)

    return TrainingResult(
        theta=model.coef_[0].copy(),
        bias=float(model.intercept_[0]),
        lambda1=lambda1,
        lambda2=lambda2,
        train_loss=train_loss,
        val_loss=train_loss,  # No validation set
        means=means.copy(),
        stds=stds.copy()
    )
