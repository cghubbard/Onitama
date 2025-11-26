"""
Card class for Onitama game.
"""
from typing import List, Tuple
from src.utils.constants import BLUE, RED


class Card:
    """
    Represents a move card in Onitama.
    
    Each card has a name and a set of relative movements that it allows.
    The movements are different depending on the player color due to perspective.
    """
    
    def __init__(self, name: str, movements: List[Tuple[int, int]]):
        """
        Initialize a card with its name and possible movements.
        
        Args:
            name: The name of the card
            movements: List of (dx, dy) tuples representing possible relative movements
                       where dx is horizontal movement and dy is vertical movement
        """
        self.name = name
        self.movements = movements
        
        # Create reversed movements for the opposing player (due to perspective)
        # Since RED sees the board from the opposite side:
        # - Up for BLUE is Down for RED (negate dy)
        # - Right for BLUE is Left for RED (negate dx)
        self.reversed_movements = [(-dx, -dy) for dx, dy in movements]
    
    def get_movements(self, player: int) -> List[Tuple[int, int]]:
        """
        Get the possible movements for a player.
        
        Args:
            player: Player color (BLUE or RED)
            
        Returns:
            List of (dx, dy) tuples for valid moves from this card,
            where dx is horizontal movement and dy is vertical movement
        """
        return self.movements if player == RED else self.reversed_movements
    
    def __str__(self) -> str:
        """Return string representation of the card."""
        return f"Card({self.name})"
    
    def __repr__(self) -> str:
        """Return string representation of the card."""
        return self.__str__()
