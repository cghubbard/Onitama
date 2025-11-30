"""
Utilities for reconstructing Game objects from StateSnapshots.

This module enables extracting features from historical game states
stored in trajectories, which is essential for training RL agents.
"""
from typing import TYPE_CHECKING

from src.game.game import Game
from src.game.card import Card
from src.game.serialization import deserialize_board
from src.utils.constants import BLUE, RED, MOVE_CARDS

if TYPE_CHECKING:
    from src.logging.trajectory import StateSnapshot


def reconstruct_game_from_snapshot(snapshot: 'StateSnapshot') -> Game:
    """
    Reconstruct a Game object from a StateSnapshot.

    Creates a Game instance matching the exact state captured in the snapshot.
    This bypasses normal initialization to avoid random card selection.

    Args:
        snapshot: StateSnapshot from a game trajectory

    Returns:
        Game instance matching the snapshot state

    Example:
        >>> from src.logging.storage import GameStorage
        >>> storage = GameStorage('data')
        >>> trajectory = storage.load_trajectory(game_id)
        >>> snapshot = trajectory.transitions[10].state
        >>> game = reconstruct_game_from_snapshot(snapshot)
        >>> # Now you can extract features from this reconstructed game

    Note:
        - The reconstructed Game bypasses __init__ to avoid randomness
        - move_history is set to empty list (not needed for feature extraction)
        - All state fields are set from the snapshot
    """
    # Create blank Game instance without calling __init__
    # This avoids random card selection and board initialization
    game = Game.__new__(Game)

    # Deserialize and set board state
    game.board = deserialize_board(snapshot.board)

    # Reconstruct Card objects from card names
    game.player_cards = {
        BLUE: [
            Card(name, MOVE_CARDS[name])
            for name in snapshot.blue_cards
        ],
        RED: [
            Card(name, MOVE_CARDS[name])
            for name in snapshot.red_cards
        ]
    }
    game.neutral_card = Card(snapshot.neutral_card, MOVE_CARDS[snapshot.neutral_card])

    # Set game state
    game.current_player = snapshot.current_player
    game.outcome = snapshot.outcome

    # Empty move history (not needed for feature extraction)
    game.move_history = []

    return game


def reconstruct_game_at_move(trajectory: 'GameTrajectory', move_number: int) -> Game:
    """
    Reconstruct the game state at a specific move in a trajectory.

    Args:
        trajectory: GameTrajectory containing the full game record
        move_number: Move number (0-indexed) to reconstruct

    Returns:
        Game instance at the specified move

    Raises:
        IndexError: If move_number is out of bounds

    Example:
        >>> game_at_move_5 = reconstruct_game_at_move(trajectory, 5)
        >>> extractor = FeatureExtractor()
        >>> features = extractor.extract(game_at_move_5, BLUE)
    """
    if not (0 <= move_number < len(trajectory.transitions)):
        raise IndexError(
            f"Move number {move_number} out of range "
            f"(trajectory has {len(trajectory.transitions)} moves)"
        )

    snapshot = trajectory.transitions[move_number].state
    return reconstruct_game_from_snapshot(snapshot)
