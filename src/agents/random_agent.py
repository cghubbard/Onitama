"""
Random Agent for Onitama game.
"""
import random
from typing import Tuple, Optional, TYPE_CHECKING
from src.agents.agent import Agent

if TYPE_CHECKING:
    from src.game.game import Game


class RandomAgent(Agent):
    """
    Agent that selects moves randomly from available legal moves.
    
    This serves as a baseline agent and can be used for testing.
    """
    
    def select_move(self, game: 'Game') -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """
        Select a random move from the available legal moves.
        
        Args:
            game: The current game state
            
        Returns:
            A tuple (from_pos, to_pos, card_name) representing the selected move,
            or None if no legal moves are available
        """
        legal_moves = game.get_legal_moves(self.player_id)
        if not legal_moves:
            return None
        
        return random.choice(legal_moves)
