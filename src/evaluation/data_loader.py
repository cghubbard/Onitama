"""
Data loading pipeline for training linear value functions.

This module loads game trajectories from storage and converts them into
training datasets with features, labels, and sample weights.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from src.logging.storage import GameStorage
from src.logging.reconstruction import reconstruct_game_from_snapshot
from src.evaluation.features import FeatureExtractor
from src.evaluation.weights import FEATURE_NAMES


@dataclass
class TrainingExample:
    """
    Single training example from a game state.

    Represents one (state, label, weight) tuple for supervised learning.
    """
    features: np.ndarray    # Raw features φ(s), shape (11,)
    label: int              # 1 if current player wins, 0 if loses
    weight: float           # Time-based weight γ^(H-1-t)
    game_id: str            # For game-level CV splitting
    move_number: int        # For debugging/analysis

    def __repr__(self):
        return (
            f"TrainingExample(game_id={self.game_id[:8]}..., "
            f"move={self.move_number}, label={self.label}, weight={self.weight:.3f})"
        )


@dataclass
class TrainingDataset:
    """
    Complete training dataset with metadata.

    Contains all training examples plus metadata needed for training
    and cross-validation.
    """
    examples: List[TrainingExample]
    game_ids: List[str]         # Unique game IDs for CV splitting
    feature_names: List[str]    # FEATURE_NAMES from weights.py
    gamma: float                # Discount factor used for weighting

    def __len__(self):
        return len(self.examples)

    def __repr__(self):
        return (
            f"TrainingDataset(n_examples={len(self.examples)}, "
            f"n_games={len(self.game_ids)}, gamma={self.gamma})"
        )

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert to numpy arrays for sklearn.

        Returns:
            X: Feature matrix, shape (n_examples, 11)
            y: Labels, shape (n_examples,)
            w: Sample weights, shape (n_examples,)
            g: Game IDs for GroupKFold, shape (n_examples,)
        """
        X = np.array([ex.features for ex in self.examples])
        y = np.array([ex.label for ex in self.examples])
        w = np.array([ex.weight for ex in self.examples])
        g = np.array([ex.game_id for ex in self.examples])
        return X, y, w, g

    def get_statistics(self) -> dict:
        """Get summary statistics of the dataset."""
        X, y, w, g = self.to_arrays()

        return {
            'n_examples': len(self.examples),
            'n_games': len(self.game_ids),
            'n_wins': int(np.sum(y)),
            'n_losses': int(len(y) - np.sum(y)),
            'win_rate': float(np.mean(y)),
            'total_weight': float(np.sum(w)),
            'mean_weight': float(np.mean(w)),
            'gamma': self.gamma,
            'feature_names': self.feature_names
        }


def load_training_data(
    storage: GameStorage,
    blue_agent: Optional[str] = None,
    red_agent: Optional[str] = None,
    limit: Optional[int] = None,
    gamma: float = 0.97,
    exclude_draws: bool = True,
    verbose: bool = False,
    agent_match_mode: str = "contains",
    terminal_weight_multiplier: float = 1.0,
    terminal_only: bool = False
) -> TrainingDataset:
    """
    Load training data from game storage.

    Loads game trajectories and converts them into training examples
    with features, labels, and time-based sample weights.

    Args:
        storage: GameStorage instance
        blue_agent: Filter by blue agent (e.g., 'heuristic')
        red_agent: Filter by red agent
        limit: Maximum number of games to load (None = all)
        gamma: Discount factor for time-based weighting (default: 0.97)
        exclude_draws: Whether to exclude drawn games (default: True)
        verbose: Print progress information (default: False)
        agent_match_mode: How to match agent names - "exact", "contains", or "prefix" (default: "contains")
        terminal_weight_multiplier: Multiply terminal state weights by this factor (default: 1.0)
        terminal_only: Only include terminal states in training data (default: False)

    Returns:
        TrainingDataset ready for training

    Process:
        1. Query games from storage (filtering by agents if specified)
        2. Load each GameTrajectory
        3. For each transition (including terminal states):
           - Reconstruct Game from StateSnapshot
           - Extract features φ(s) from current player's perspective
           - Determine label: 1 if current player wins, 0 if loses
           - Compute weight: γ^(H-1-t)
        4. Collect all examples into TrainingDataset

    Example:
        >>> from src.logging.storage import GameStorage
        >>> storage = GameStorage('data')
        >>> dataset = load_training_data(
        ...     storage,
        ...     blue_agent='heuristic',
        ...     red_agent='heuristic',
        ...     limit=1000
        ... )
        >>> print(f"Loaded {len(dataset)} examples from {len(dataset.game_ids)} games")
    """
    examples = []
    game_ids_set = set()

    # Query games from storage
    if verbose:
        print(f"Querying games...")
        print(f"  Filters: blue_agent={blue_agent}, red_agent={red_agent}")
        print(f"  Match mode: {agent_match_mode}")
        print(f"  Limit: {limit if limit else 'all'}")

        # Show available matchups if filtering
        if blue_agent or red_agent:
            combos = storage.get_unique_agent_combinations()
            print(f"\n  Available agent matchups (top 10):")
            for blue, red, count in combos[:10]:
                print(f"    {blue} vs {red}: {count} games")

    games_metadata = storage.query_games(
        blue_agent=blue_agent,
        red_agent=red_agent,
        limit=limit or 10000,  # Large default if no limit specified
        agent_match_mode=agent_match_mode
    )

    # Validate filters matched games
    if (blue_agent or red_agent) and len(games_metadata) == 0:
        total_games = storage.count_games()
        combos = storage.get_unique_agent_combinations()

        error_msg = (
            f"\nNo games found matching filters:\n"
            f"  blue_agent={blue_agent}\n"
            f"  red_agent={red_agent}\n"
            f"  match_mode={agent_match_mode}\n"
            f"\nDatabase contains {total_games} total games.\n"
            f"\nAvailable matchups:\n"
        )
        for blue, red, count in combos[:15]:
            error_msg += f"  {blue} vs {red}: {count} games\n"

        raise ValueError(error_msg)

    if verbose:
        print(f"Found {len(games_metadata)} games matching criteria")

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Process each game
    n_games_processed = 0
    n_games_excluded = 0

    for game_meta in games_metadata:
        # Load full trajectory
        trajectory = storage.load_trajectory(game_meta['game_id'])

        # Skip draws if requested
        if exclude_draws and trajectory.outcome.winner is None:
            n_games_excluded += 1
            continue

        winner = trajectory.outcome.winner  # 0=BLUE, 1=RED, or None
        H = len(trajectory.transitions)     # Game length

        game_ids_set.add(trajectory.game_id)
        n_games_processed += 1

        # Process each transition (including terminal states)
        for t, transition in enumerate(trajectory.transitions):
            snapshot = transition.state

            # Reconstruct game from snapshot
            game = reconstruct_game_from_snapshot(snapshot)

            # Compute time-based weight: γ^(H-1-t)
            # Later states (higher t) get higher weights (closer to outcome)
            weight = gamma ** (H - 1 - t)

            # For terminal states, create examples from BOTH players' perspectives
            # This ensures we learn what winning AND losing positions look like
            from src.utils.constants import ONGOING
            if snapshot.outcome != ONGOING:
                # Terminal state - create two examples with boosted weight
                terminal_weight = weight * terminal_weight_multiplier
                for player_id in [0, 1]:  # BLUE=0, RED=1
                    features = extractor.extract_as_array(game, player_id)
                    label = 1 if winner == player_id else 0

                    examples.append(TrainingExample(
                        features=np.array(features, dtype=np.float64),
                        label=label,
                        weight=terminal_weight,
                        game_id=trajectory.game_id,
                        move_number=t
                    ))
            elif not terminal_only:
                # Non-terminal state - create one example from current player's perspective
                # Skip if terminal_only=True
                current_player = snapshot.current_player
                features = extractor.extract_as_array(game, current_player)
                label = 1 if winner == current_player else 0

                examples.append(TrainingExample(
                    features=np.array(features, dtype=np.float64),
                    label=label,
                    weight=weight,
                    game_id=trajectory.game_id,
                    move_number=t
                ))

        # Progress update
        if verbose and n_games_processed % 100 == 0:
            print(f"  Processed {n_games_processed}/{len(games_metadata)} games, "
                  f"{len(examples)} examples so far...")

    if verbose:
        print(f"\nData loading complete:")
        print(f"  Games processed: {n_games_processed}")
        if exclude_draws:
            print(f"  Draws excluded: {n_games_excluded}")
        print(f"  Total examples: {len(examples)}")
        print(f"  Avg examples/game: {len(examples)/n_games_processed:.1f}")

    return TrainingDataset(
        examples=examples,
        game_ids=sorted(game_ids_set),
        feature_names=FEATURE_NAMES.copy(),
        gamma=gamma
    )


def split_by_games(
    dataset: TrainingDataset,
    train_games: List[str],
    val_games: List[str]
) -> Tuple[TrainingDataset, TrainingDataset]:
    """
    Split dataset into train and validation sets by game ID.

    Ensures that all states from a game stay together in the same split.

    Args:
        dataset: Complete training dataset
        train_games: List of game IDs for training
        val_games: List of game IDs for validation

    Returns:
        Tuple of (train_dataset, val_dataset)

    Example:
        >>> from sklearn.model_selection import train_test_split
        >>> train_games, val_games = train_test_split(
        ...     dataset.game_ids,
        ...     test_size=0.2,
        ...     random_state=42
        ... )
        >>> train_ds, val_ds = split_by_games(dataset, train_games, val_games)
    """
    train_games_set = set(train_games)
    val_games_set = set(val_games)

    train_examples = [ex for ex in dataset.examples if ex.game_id in train_games_set]
    val_examples = [ex for ex in dataset.examples if ex.game_id in val_games_set]

    train_dataset = TrainingDataset(
        examples=train_examples,
        game_ids=[gid for gid in dataset.game_ids if gid in train_games_set],
        feature_names=dataset.feature_names.copy(),
        gamma=dataset.gamma
    )

    val_dataset = TrainingDataset(
        examples=val_examples,
        game_ids=[gid for gid in dataset.game_ids if gid in val_games_set],
        feature_names=dataset.feature_names.copy(),
        gamma=dataset.gamma
    )

    return train_dataset, val_dataset
