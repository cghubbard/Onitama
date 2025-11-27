"""
Linear Heuristic Agent for Onitama game.

Uses a linear evaluation function V(s) = w^T * φ(s) to evaluate board positions
and select moves that lead to the best resulting position.
"""

import random
from typing import Tuple, Optional, List, TYPE_CHECKING

from src.agents.agent import Agent
from src.evaluation.features import FeatureExtractor
from src.evaluation.weights import DEFAULT_WEIGHT_VECTOR
from src.utils.constants import ONGOING

if TYPE_CHECKING:
    from src.game.game import Game


class LinearHeuristicAgent(Agent):
    """
    Agent that uses a linear heuristic evaluation of positions.

    Evaluates board positions using a feature-based linear function:
    V(s) = w^T * φ(s)

    where φ(s) is a vector of 11 features and w is a weight vector.

    Features include:
    - Material balance (student count difference)
    - Master survival indicators
    - Master safety (threat balance)
    - Mobility (legal moves difference)
    - Capture opportunities
    - Temple distance progress
    - Student advancement
    - Central control
    - Card mobility
    - Master escape options
    """

    def __init__(
        self,
        player_id: int,
        weights: Optional[List[float]] = None,
        randomize: bool = True
    ):
        """
        Initialize a linear heuristic agent.

        Args:
            player_id: The player ID (BLUE or RED) this agent controls
            weights: Custom weight vector (11 floats) or None for defaults
            randomize: Whether to add randomization to break ties
        """
        super().__init__(player_id)
        self.weights = weights if weights is not None else DEFAULT_WEIGHT_VECTOR
        self.randomize = randomize
        self.extractor = FeatureExtractor()

    def select_move(
        self,
        game: 'Game'
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """
        Select the best move based on position evaluation.

        Evaluates each legal move by simulating it and evaluating the
        resulting position using the linear heuristic function.

        Args:
            game: The current game state

        Returns:
            A tuple (from_pos, to_pos, card_name) representing the selected move,
            or None if no legal moves are available
        """
        legal_moves = game.get_legal_moves(self.player_id)
        if not legal_moves:
            return None

        # Evaluate each move
        move_scores = []
        for move in legal_moves:
            score = self._evaluate_move(game, move)
            move_scores.append((move, score))

        # Find the best move(s)
        max_score = max(score for _, score in move_scores)
        best_moves = [move for move, score in move_scores if score == max_score]

        # Either choose randomly among the best moves or just take the first one
        if self.randomize and len(best_moves) > 1:
            return random.choice(best_moves)
        else:
            return best_moves[0]

    def _evaluate_move(
        self,
        game: 'Game',
        move: Tuple[Tuple[int, int], Tuple[int, int], str]
    ) -> float:
        """
        Evaluate a move by simulating it and evaluating the resulting position.

        Args:
            game: The current game state
            move: A tuple (from_pos, to_pos, card_name) representing the move

        Returns:
            A score for the move (higher is better)
        """
        from_pos, to_pos, card_name = move

        # Create a copy of the game to simulate the move
        simulated = game.copy()
        simulated.make_move(from_pos, to_pos, card_name)

        # Check for immediate win/loss
        outcome = simulated.get_outcome()
        if outcome != ONGOING:
            # Check if we won
            from src.utils.constants import BLUE_WINS, RED_WINS, BLUE
            if self.player_id == BLUE:
                if outcome == BLUE_WINS:
                    return float('inf')
                elif outcome == RED_WINS:
                    return float('-inf')
            else:
                if outcome == RED_WINS:
                    return float('inf')
                elif outcome == BLUE_WINS:
                    return float('-inf')
            # Draw
            return 0.0

        # Evaluate position from our perspective
        return self.extractor.evaluate(simulated, self.player_id, self.weights)

    def get_weights(self) -> List[float]:
        """
        Get the current weight vector.

        Returns:
            List of 11 weights
        """
        return self.weights.copy()

    def set_weights(self, weights: List[float]) -> None:
        """
        Set new weights for the evaluation function.

        Args:
            weights: List of 11 weights
        """
        if len(weights) != 11:
            raise ValueError(f"Expected 11 weights, got {len(weights)}")
        self.weights = weights.copy()
