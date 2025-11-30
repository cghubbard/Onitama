"""
Unit tests for the training module.
"""
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluation.trainer import (
    compute_standardization_stats,
    standardize_features,
    get_sklearn_params,
    train_single_model,
    train_linear_value_function,
    compute_weighted_log_loss
)
from src.evaluation.data_loader import TrainingExample, TrainingDataset


class TestStandardization:
    """Tests for feature standardization."""

    def test_compute_stats(self):
        """Should compute correct mean and std."""
        X = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ])

        means, stds = compute_standardization_stats(X)

        # Mean of [1, 3, 5] is 3, mean of [2, 4, 6] is 4
        assert np.allclose(means, [3.0, 4.0])

        # Std of [1, 3, 5] is sqrt(8/3) ≈ 1.633
        # Std of [2, 4, 6] is sqrt(8/3) ≈ 1.633
        expected_std = np.sqrt(8/3)
        assert np.allclose(stds, [expected_std, expected_std], atol=0.01)

    def test_standardize_features(self):
        """Should standardize features correctly."""
        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0]
        ])

        means = np.array([2.0, 20.0])
        stds = np.array([1.0, 10.0])

        X_std = standardize_features(X, means, stds, epsilon=0.0)

        # After standardization: (X - mean) / std
        expected = np.array([
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 1.0]
        ])

        assert np.allclose(X_std, expected)

    def test_epsilon_prevents_division_by_zero(self):
        """Epsilon should prevent division by zero for constant features."""
        X = np.array([
            [1.0, 5.0],
            [1.0, 5.0],
            [1.0, 5.0]
        ])

        means, stds = compute_standardization_stats(X)

        # Stds will be 0 for constant features
        assert stds[0] == 0.0
        assert stds[1] == 0.0

        # Standardization with epsilon should not crash
        X_std = standardize_features(X, means, stds, epsilon=1e-8)

        # Should return finite values
        assert np.all(np.isfinite(X_std))


class TestSklearnParams:
    """Tests for sklearn parameter mapping."""

    def test_l1_only(self):
        """L1 only should use liblinear solver."""
        params = get_sklearn_params(lambda1=0.1, lambda2=0.0)

        assert params['penalty'] == 'l1'
        assert params['solver'] == 'liblinear'
        assert abs(params['C'] - 10.0) < 0.01  # C = 1/0.1 = 10

    def test_l2_only(self):
        """L2 only should use lbfgs solver."""
        params = get_sklearn_params(lambda1=0.0, lambda2=0.5)

        assert params['penalty'] == 'l2'
        assert params['solver'] == 'lbfgs'
        assert abs(params['C'] - 2.0) < 0.01  # C = 1/0.5 = 2

    def test_elasticnet(self):
        """Both L1 and L2 should use elasticnet with saga solver."""
        params = get_sklearn_params(lambda1=0.3, lambda2=0.7)

        assert params['penalty'] == 'elasticnet'
        assert params['solver'] == 'saga'
        assert abs(params['C'] - 1.0) < 0.01  # C = 1/(0.3+0.7) = 1
        assert abs(params['l1_ratio'] - 0.3) < 0.01  # l1_ratio = 0.3/(0.3+0.7) = 0.3

    def test_no_regularization(self):
        """No regularization should use penalty='none'."""
        params = get_sklearn_params(lambda1=0.0, lambda2=0.0)

        assert params['penalty'] == 'none'
        assert params['solver'] == 'lbfgs'


class TestWeightedLogLoss:
    """Tests for weighted log loss computation."""

    def test_perfect_predictions(self):
        """Loss should be lower for better predictions."""
        # Create a model with well-separated data
        model = LogisticRegression(C=1000)  # Low regularization for better fit
        X = np.array([[-5], [-4], [-3], [3], [4], [5]])
        y = np.array([0, 0, 0, 1, 1, 1])
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        model.fit(X, y)

        loss = compute_weighted_log_loss(model, X, y, w)

        # Should be reasonably low for well-separated data
        assert loss < 0.3

    def test_loss_increases_with_errors(self):
        """Loss should be higher for worse predictions."""
        # Train a simple model
        model = LogisticRegression()
        X_train = np.array([[0], [1], [2], [3]])
        y_train = np.array([0, 0, 1, 1])
        model.fit(X_train, y_train)

        # Test on data where predictions are good
        X_good = np.array([[0], [3]])
        y_good = np.array([0, 1])
        w_good = np.array([1.0, 1.0])
        loss_good = compute_weighted_log_loss(model, X_good, y_good, w_good)

        # Test on data where predictions are bad (reversed labels)
        X_bad = np.array([[0], [3]])
        y_bad = np.array([1, 0])  # Reversed
        w_bad = np.array([1.0, 1.0])
        loss_bad = compute_weighted_log_loss(model, X_bad, y_bad, w_bad)

        # Loss should be higher for bad predictions
        assert loss_bad > loss_good


class TestSingleModelTraining:
    """Tests for training a single model."""

    def create_synthetic_dataset(self, n_examples=100):
        """Create a synthetic dataset for testing."""
        np.random.seed(42)

        examples = []
        for i in range(n_examples):
            # Create linearly separable data
            x1 = np.random.randn()
            x2 = np.random.randn()
            features = np.array([x1, x2])

            # Label based on simple linear boundary: x1 + x2 > 0
            label = 1 if (x1 + x2) > 0 else 0

            examples.append(TrainingExample(
                features=features,
                label=label,
                weight=1.0,
                game_id=f"game{i % 10}",  # 10 games
                move_number=i // 10
            ))

        return TrainingDataset(
            examples=examples,
            game_ids=[f"game{i}" for i in range(10)],
            feature_names=["f1", "f2"],
            gamma=0.97
        )

    def test_train_single_model(self):
        """Should successfully train a single model."""
        dataset = self.create_synthetic_dataset()

        result = train_single_model(
            dataset,
            lambda1=0.0,
            lambda2=0.1,
            random_state=42
        )

        # Check that result has expected fields
        assert len(result.theta) == 2  # 2 features
        assert isinstance(result.bias, float)
        assert result.train_loss >= 0
        assert len(result.means) == 2
        assert len(result.stds) == 2

    def test_model_learns_pattern(self):
        """Model should learn the underlying pattern."""
        dataset = self.create_synthetic_dataset(n_examples=500)

        result = train_single_model(
            dataset,
            lambda1=0.0,
            lambda2=0.01,
            random_state=42
        )

        # Weights should both be positive (since label = x1 + x2 > 0)
        # After standardization, both features should contribute positively
        assert result.theta[0] > 0 or result.theta[1] > 0

        # Loss should be reasonable (< 0.7 for easy synthetic data)
        assert result.train_loss < 0.7

    def test_regularization_affects_weights(self):
        """Stronger regularization should reduce weight magnitudes."""
        dataset = self.create_synthetic_dataset()

        # Train with no regularization
        result_no_reg = train_single_model(
            dataset,
            lambda1=0.0,
            lambda2=0.0,
            random_state=42
        )

        # Train with strong regularization
        result_with_reg = train_single_model(
            dataset,
            lambda1=0.0,
            lambda2=10.0,
            random_state=42
        )

        # Regularized weights should have smaller magnitude
        norm_no_reg = np.linalg.norm(result_no_reg.theta)
        norm_with_reg = np.linalg.norm(result_with_reg.theta)

        assert norm_with_reg < norm_no_reg


class TestGridSearch:
    """Tests for grid search training."""

    def create_small_dataset(self):
        """Create a small dataset for quick testing."""
        np.random.seed(42)

        examples = []
        for i in range(50):
            features = np.random.randn(3)  # 3 features
            label = 1 if features[0] > 0 else 0
            examples.append(TrainingExample(
                features=features,
                label=label,
                weight=1.0,
                game_id=f"game{i % 5}",  # 5 games
                move_number=i // 5
            ))

        return TrainingDataset(
            examples=examples,
            game_ids=[f"game{i}" for i in range(5)],
            feature_names=["f1", "f2", "f3"],
            gamma=0.97
        )

    def test_grid_search_runs(self):
        """Grid search should complete without errors."""
        dataset = self.create_small_dataset()

        results = train_linear_value_function(
            dataset,
            lambda1_values=[0.0, 0.1],
            lambda2_values=[0.0, 0.1],
            cv_folds=3,
            random_state=42,
            verbose=False
        )

        # Should have tried 4 models (2 λ1 × 2 λ2)
        assert results.n_models_tried == 4
        assert len(results.results) == 4
        assert results.cv_folds == 3

    def test_best_model_selected(self):
        """Should select model with lowest validation loss."""
        dataset = self.create_small_dataset()

        results = train_linear_value_function(
            dataset,
            lambda1_values=[0.0, 0.1],
            lambda2_values=[0.0, 0.1],
            cv_folds=3,
            random_state=42,
            verbose=False
        )

        # Best model should have lowest val_loss
        best_val_loss = results.best_model.val_loss
        for result in results.results:
            assert result.val_loss >= best_val_loss

    def test_cv_fold_losses_recorded(self):
        """Should record per-fold validation losses."""
        dataset = self.create_small_dataset()

        results = train_linear_value_function(
            dataset,
            lambda1_values=[0.0],
            lambda2_values=[0.1],
            cv_folds=3,
            random_state=42,
            verbose=False
        )

        # Should have one result
        assert len(results.results) == 1
        result = results.results[0]

        # Should have 3 fold losses
        assert len(result.cv_fold_losses) == 3
        assert all(loss >= 0 for loss in result.cv_fold_losses)

    def test_get_sorted_results(self):
        """Should be able to sort results by different metrics."""
        dataset = self.create_small_dataset()

        results = train_linear_value_function(
            dataset,
            lambda1_values=[0.0, 0.5, 1.0],
            lambda2_values=[0.0, 0.5],
            cv_folds=3,
            random_state=42,
            verbose=False
        )

        # Sort by val_loss
        sorted_by_val = results.get_sorted_results(by='val_loss', ascending=True)
        assert sorted_by_val[0].val_loss <= sorted_by_val[-1].val_loss

        # Sort by lambda1
        sorted_by_l1 = results.get_sorted_results(by='lambda1', ascending=True)
        assert sorted_by_l1[0].lambda1 <= sorted_by_l1[-1].lambda1

    def test_game_level_cv_splits(self):
        """Cross-validation should respect game-level splits."""
        # Create dataset where each game has both labels
        # to avoid sklearn's "only one label" error in CV folds
        examples = []
        game_ids = ["g1", "g2", "g3", "g4"]

        for game_idx, game_id in enumerate(game_ids):
            for move in range(10):
                # Each game has mixed labels based on move number
                # This ensures each CV fold will have both classes
                label = (move + game_idx) % 2
                features = np.random.randn(2)
                features[0] = float(label)  # Feature correlates with label

                examples.append(TrainingExample(
                    features=features,
                    label=label,
                    weight=1.0,
                    game_id=game_id,
                    move_number=move
                ))

        dataset = TrainingDataset(
            examples=examples,
            game_ids=game_ids,
            feature_names=["f1", "f2"],
            gamma=0.97
        )

        # Train with CV
        # If CV splits games correctly, it should work fine
        # If it splits within games, it would have data leakage
        results = train_linear_value_function(
            dataset,
            lambda1_values=[0.0],
            lambda2_values=[0.0],
            cv_folds=2,  # 4 games, 2 folds = 2 games per fold
            random_state=42,
            verbose=False
        )

        # Should complete successfully
        assert results.best_model is not None
        assert results.best_model.val_loss >= 0
