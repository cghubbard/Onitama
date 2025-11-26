"""
Base Agent class for Onitama.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.game import Game


class Agent(ABC):
    """
    Abstract base class for Onitama agents.
    
    All agents must implement the select_move method to choose a move
    from the available legal moves.
    """
    
    def __init__(self, player_id: int):
        """
        Initialize an agent.
        
        Args:
            player_id: The player ID (BLUE or RED) this agent controls
        """
        self.player_id = player_id
    
    @abstractmethod
    def select_move(self, game: 'Game') -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """
        Select a move from the available legal moves.
        
        Args:
            game: The current game state
            
        Returns:
            A tuple (from_pos, to_pos, card_name) representing the selected move,
            or None if no legal moves are available
        """
        pass
