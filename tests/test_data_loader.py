"""
Unit tests for the data loading pipeline.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from src.logging.storage import GameStorage
from src.logging.trajectory import GameTrajectory, GameConfig, Transition, StateSnapshot
from src.evaluation.data_loader import (
    load_training_data,
    TrainingExample,
    TrainingDataset,
    split_by_games
)
from src.utils.constants import BLUE, RED, PAWN, MASTER, ONGOING, BLUE_WINS, RED_WINS


class TestTrainingDataStructures:
    """Tests for data structures."""

    def test_training_example_creation(self):
        """Should create TrainingExample with correct fields."""
        features = np.array([1.0, 2.0, 3.0])
        example = TrainingExample(
            features=features,
            label=1,
            weight=0.97,
            game_id="test-123",
            move_number=5
        )

        assert np.array_equal(example.features, features)
        assert example.label == 1
        assert example.weight == 0.97
        assert example.game_id == "test-123"
        assert example.move_number == 5

    def test_training_dataset_to_arrays(self):
        """Should convert TrainingDataset to numpy arrays."""
        examples = [
            TrainingExample(
                features=np.array([1.0, 2.0]),
                label=1,
                weight=0.97,
                game_id="game1",
                move_number=0
            ),
            TrainingExample(
                features=np.array([3.0, 4.0]),
                label=0,
                weight=0.94,
                game_id="game1",
                move_number=1
            ),
            TrainingExample(
                features=np.array([5.0, 6.0]),
                label=1,
                weight=1.0,
                game_id="game2",
                move_number=0
            )
        ]

        dataset = TrainingDataset(
            examples=examples,
            game_ids=["game1", "game2"],
            feature_names=["feat1", "feat2"],
            gamma=0.97
        )

        X, y, w, g = dataset.to_arrays()

        # Check shapes
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        assert w.shape == (3,)
        assert g.shape == (3,)

        # Check values
        assert np.array_equal(X, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        assert np.array_equal(y, np.array([1, 0, 1]))
        assert np.allclose(w, np.array([0.97, 0.94, 1.0]))
        assert np.array_equal(g, np.array(["game1", "game1", "game2"]))

    def test_dataset_statistics(self):
        """Should compute correct statistics."""
        examples = [
            TrainingExample(
                features=np.array([1.0]),
                label=1,
                weight=1.0,
                game_id="g1",
                move_number=0
            ),
            TrainingExample(
                features=np.array([2.0]),
                label=0,
                weight=0.5,
                game_id="g1",
                move_number=1
            ),
            TrainingExample(
                features=np.array([3.0]),
                label=1,
                weight=0.25,
                game_id="g2",
                move_number=0
            )
        ]

        dataset = TrainingDataset(
            examples=examples,
            game_ids=["g1", "g2"],
            feature_names=["f1"],
            gamma=0.5
        )

        stats = dataset.get_statistics()

        assert stats['n_examples'] == 3
        assert stats['n_games'] == 2
        assert stats['n_wins'] == 2
        assert stats['n_losses'] == 1
        assert abs(stats['win_rate'] - 2/3) < 0.01
        assert abs(stats['total_weight'] - 1.75) < 0.01
        assert abs(stats['mean_weight'] - 1.75/3) < 0.01


class TestDataLoader:
    """Tests for the data loading pipeline."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary game storage for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = GameStorage(data_dir=temp_dir)
        yield storage
        shutil.rmtree(temp_dir)

    def create_test_trajectory(
        self,
        game_id: str,
        winner: int,
        num_moves: int = 3,
        blue_agent: str = "test",
        red_agent: str = "test"
    ) -> GameTrajectory:
        """Helper to create a test trajectory."""
        trajectory = GameTrajectory(
            game_id=game_id,
            timestamp="2025-01-01T00:00:00Z",
            config=GameConfig(
                cards_used=["Tiger", "Dragon", "Frog", "Rabbit", "Crab"],
                blue_agent=blue_agent,
                red_agent=red_agent
            )
        )

        # Add transitions
        for move in range(num_moves):
            snapshot = StateSnapshot(
                board={
                    "2,0": [BLUE, MASTER],
                    "1,0": [BLUE, PAWN],
                    "2,4": [RED, MASTER],
                    "3,4": [RED, PAWN],
                },
                current_player=BLUE if move % 2 == 0 else RED,
                blue_cards=["Tiger", "Dragon"],
                red_cards=["Frog", "Rabbit"],
                neutral_card="Crab",
                move_number=move,
                outcome=ONGOING
            )

            transition = Transition(
                move_number=move,
                state=snapshot,
                legal_moves=[],
                action={"from": [2, 0], "to": [2, 1], "card": "Tiger"}
            )
            trajectory.add_transition(transition)

        # Set outcome
        trajectory.set_outcome(winner, "test")

        return trajectory

    def test_load_single_game(self, temp_storage):
        """Should load training data from a single game."""
        # Create and save a game
        trajectory = self.create_test_trajectory(
            game_id="game1",
            winner=BLUE,
            num_moves=3
        )
        temp_storage.save_trajectory(trajectory)

        # Load training data
        dataset = load_training_data(temp_storage, gamma=0.97)

        # Should have 3 examples (one per move)
        assert len(dataset) == 3
        assert len(dataset.game_ids) == 1
        assert dataset.game_ids[0] == "game1"
        assert dataset.gamma == 0.97

    def test_label_assignment_for_winner(self, temp_storage):
        """Labels should be 1 when current player wins."""
        # BLUE wins the game
        trajectory = self.create_test_trajectory(
            game_id="game1",
            winner=BLUE,
            num_moves=4  # moves: BLUE, RED, BLUE, RED
        )
        temp_storage.save_trajectory(trajectory)

        dataset = load_training_data(temp_storage)

        # Check labels
        for i, example in enumerate(dataset.examples):
            if i % 2 == 0:  # BLUE's turn
                assert example.label == 1  # BLUE wins, so label=1
            else:  # RED's turn
                assert example.label == 0  # BLUE wins, so RED loses, label=0

    def test_label_assignment_for_loser(self, temp_storage):
        """Labels should be 0 when current player loses."""
        # RED wins the game
        trajectory = self.create_test_trajectory(
            game_id="game1",
            winner=RED,
            num_moves=4
        )
        temp_storage.save_trajectory(trajectory)

        dataset = load_training_data(temp_storage)

        # Check labels
        for i, example in enumerate(dataset.examples):
            if i % 2 == 0:  # BLUE's turn
                assert example.label == 0  # RED wins, so BLUE loses, label=0
            else:  # RED's turn
                assert example.label == 1  # RED wins, so label=1

    def test_weight_computation(self, temp_storage):
        """Weights should be γ^(H-1-t)."""
        trajectory = self.create_test_trajectory(
            game_id="game1",
            winner=BLUE,
            num_moves=3  # H = 3
        )
        temp_storage.save_trajectory(trajectory)

        gamma = 0.97
        dataset = load_training_data(temp_storage, gamma=gamma)

        # H = 3, so weights should be:
        # t=0: γ^(3-1-0) = γ^2 = 0.9409
        # t=1: γ^(3-1-1) = γ^1 = 0.97
        # t=2: γ^(3-1-2) = γ^0 = 1.0

        assert len(dataset.examples) == 3
        assert abs(dataset.examples[0].weight - gamma**2) < 0.0001
        assert abs(dataset.examples[1].weight - gamma**1) < 0.0001
        assert abs(dataset.examples[2].weight - gamma**0) < 0.0001

    def test_load_multiple_games(self, temp_storage):
        """Should load data from multiple games."""
        # Create 3 games
        for i in range(3):
            trajectory = self.create_test_trajectory(
                game_id=f"game{i}",
                winner=BLUE if i % 2 == 0 else RED,
                num_moves=2
            )
            temp_storage.save_trajectory(trajectory)

        dataset = load_training_data(temp_storage)

        # Should have 6 examples total (2 per game)
        assert len(dataset) == 6
        assert len(dataset.game_ids) == 3

    def test_draw_exclusion(self, temp_storage):
        """Should exclude drawn games when requested."""
        # Create a drawn game
        trajectory = self.create_test_trajectory(
            game_id="game1",
            winner=None,  # Draw
            num_moves=2
        )
        temp_storage.save_trajectory(trajectory)

        # Create a non-drawn game
        trajectory2 = self.create_test_trajectory(
            game_id="game2",
            winner=BLUE,
            num_moves=2
        )
        temp_storage.save_trajectory(trajectory2)

        # Load with exclude_draws=True (default)
        dataset = load_training_data(temp_storage, exclude_draws=True)

        # Should only have examples from game2
        assert len(dataset) == 2
        assert len(dataset.game_ids) == 1
        assert dataset.game_ids[0] == "game2"

    def test_draw_inclusion(self, temp_storage):
        """Should include drawn games when exclude_draws=False."""
        # Create a drawn game
        trajectory = self.create_test_trajectory(
            game_id="game1",
            winner=None,  # Draw
            num_moves=2
        )
        temp_storage.save_trajectory(trajectory)

        # Load with exclude_draws=False
        dataset = load_training_data(temp_storage, exclude_draws=False)

        # Should have examples from the drawn game
        # Note: labels will both be 0 since neither player won
        assert len(dataset) == 2
        assert len(dataset.game_ids) == 1

    def test_filter_by_agent_type(self, temp_storage):
        """Should filter games by agent type."""
        # Create games with different agents
        traj1 = self.create_test_trajectory(
            game_id="game1",
            winner=BLUE,
            num_moves=2,
            blue_agent="heuristic",
            red_agent="random"
        )
        temp_storage.save_trajectory(traj1)

        traj2 = self.create_test_trajectory(
            game_id="game2",
            winner=RED,
            num_moves=2,
            blue_agent="heuristic",
            red_agent="heuristic"
        )
        temp_storage.save_trajectory(traj2)

        # Load only heuristic vs heuristic games
        dataset = load_training_data(
            temp_storage,
            blue_agent="heuristic",
            red_agent="heuristic"
        )

        # Should only have game2
        assert len(dataset) == 2
        assert len(dataset.game_ids) == 1
        assert dataset.game_ids[0] == "game2"

    def test_limit_parameter(self, temp_storage):
        """Should respect limit parameter."""
        # Create 5 games
        for i in range(5):
            trajectory = self.create_test_trajectory(
                game_id=f"game{i}",
                winner=BLUE,
                num_moves=2
            )
            temp_storage.save_trajectory(trajectory)

        # Load with limit=3
        dataset = load_training_data(temp_storage, limit=3)

        # Should have at most 3 games
        assert len(dataset.game_ids) <= 3

    def test_game_grouping_for_cv(self, temp_storage):
        """Game IDs should be preserved for GroupKFold CV."""
        # Create games with different numbers of moves
        traj1 = self.create_test_trajectory("game1", BLUE, num_moves=2)
        traj2 = self.create_test_trajectory("game2", RED, num_moves=3)
        temp_storage.save_trajectory(traj1)
        temp_storage.save_trajectory(traj2)

        dataset = load_training_data(temp_storage)

        X, y, w, g = dataset.to_arrays()

        # Check that game IDs are preserved
        # game1 has 2 examples, game2 has 3 examples
        assert len(g) == 5
        assert sum(1 for gid in g if gid == "game1") == 2
        assert sum(1 for gid in g if gid == "game2") == 3

    def test_features_are_valid(self, temp_storage):
        """Features should be valid numpy arrays."""
        trajectory = self.create_test_trajectory("game1", BLUE, num_moves=2)
        temp_storage.save_trajectory(trajectory)

        dataset = load_training_data(temp_storage)

        for example in dataset.examples:
            # Features should be numpy array
            assert isinstance(example.features, np.ndarray)
            # Should have 11 features
            assert len(example.features) == 11
            # Should be numeric
            assert example.features.dtype in [np.float32, np.float64]


class TestSplitByGames:
    """Tests for game-level dataset splitting."""

    def test_split_by_games(self):
        """Should split dataset by game IDs."""
        # Create examples from 3 games
        examples = []
        for game_id in ["g1", "g2", "g3"]:
            for move in range(2):
                examples.append(TrainingExample(
                    features=np.array([1.0]),
                    label=1,
                    weight=1.0,
                    game_id=game_id,
                    move_number=move
                ))

        dataset = TrainingDataset(
            examples=examples,
            game_ids=["g1", "g2", "g3"],
            feature_names=["f1"],
            gamma=0.97
        )

        # Split: g1, g2 for train, g3 for val
        train_ds, val_ds = split_by_games(
            dataset,
            train_games=["g1", "g2"],
            val_games=["g3"]
        )

        # Check train set
        assert len(train_ds) == 4  # 2 games × 2 moves
        assert len(train_ds.game_ids) == 2
        assert set(train_ds.game_ids) == {"g1", "g2"}

        # Check val set
        assert len(val_ds) == 2  # 1 game × 2 moves
        assert len(val_ds.game_ids) == 1
        assert val_ds.game_ids[0] == "g3"

    def test_split_preserves_game_grouping(self):
        """All examples from a game should stay in the same split."""
        examples = []
        for move in range(5):
            examples.append(TrainingExample(
                features=np.array([float(move)]),
                label=1 if move < 3 else 0,
                weight=0.97 ** (5 - move),
                game_id="game1",
                move_number=move
            ))

        dataset = TrainingDataset(
            examples=examples,
            game_ids=["game1"],
            feature_names=["f1"],
            gamma=0.97
        )

        train_ds, val_ds = split_by_games(
            dataset,
            train_games=["game1"],
            val_games=[]
        )

        # All examples should be in train
        assert len(train_ds) == 5
        assert len(val_ds) == 0
