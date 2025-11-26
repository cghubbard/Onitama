"""
Heuristic Agent for Onitama game.
"""
import random
from typing import Tuple, Optional, Dict, List, TYPE_CHECKING
import math

from src.agents.agent import Agent

if TYPE_CHECKING:
    from src.game.game import Game
from src.utils.constants import BOARD_SIZE, BLUE, RED, PAWN, MASTER, BLUE_SHRINE, RED_SHRINE


class HeuristicAgent(Agent):
    """
    Agent that uses heuristics to select moves.
    
    This agent evaluates moves based on simple heuristics such as:
    - Piece count
    - Master safety
    - Center control
    - Distance to opponent's shrine
    - Attacking opportunities
    """
    
    def __init__(self, player_id: int, randomize: bool = True):
        """
        Initialize a heuristic agent.
        
        Args:
            player_id: The player ID (BLUE or RED) this agent controls
            randomize: Whether to add randomization to break ties
        """
        super().__init__(player_id)
        self.randomize = randomize
        
        # Define weights for different heuristics
        self.weights = {
            'piece_count': 5.0,          # Having more pieces is good
            'master_safety': 3.0,        # Keeping master safe is important
            'center_control': 1.0,       # Controlling the center is advantageous
            'shrine_distance': 2.0,      # Getting master closer to opponent's shrine
            'attack_opportunity': 1.5,   # Ability to capture opponent pieces
            'master_capture': 100.0,     # Huge bonus for capturing opponent's master
            'master_to_shrine': 100.0,   # Huge bonus for moving master to opponent's shrine
        }
    
    def select_move(self, game: 'Game') -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """
        Select the best move based on heuristic evaluation.
        
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
        move_scores = {}
        for move in legal_moves:
            score = self._evaluate_move(game, move)
            move_scores[move] = score
        
        # Find the best move(s)
        max_score = max(move_scores.values())
        best_moves = [move for move, score in move_scores.items() if score == max_score]
        
        # Either choose randomly among the best moves or just take the first one
        if self.randomize and len(best_moves) > 1:
            return random.choice(best_moves)
        else:
            return best_moves[0]
    
    def _evaluate_move(self, game: 'Game', move: Tuple[Tuple[int, int], Tuple[int, int], str]) -> float:
        """
        Evaluate a potential move.
        
        Args:
            game: The current game state
            move: A tuple (from_pos, to_pos, card_name) representing the move to evaluate
            
        Returns:
            A score for the move, higher is better
        """
        from_pos, to_pos, card_name = move
        
        # Create a copy of the game to simulate the move
        # Since making a deep copy of the game might be complex,
        # we'll evaluate based on the current state and the move itself
        
        # Get the current board state
        board = game.get_board_state()
        opponent = RED if self.player_id == BLUE else BLUE
        
        # Initialize score
        score = 0.0
        
        # Check if moving piece is a master
        moving_piece = board[from_pos]
        is_master = moving_piece[1] == MASTER
        
        # 1. Piece count
        # We gain a point if we capture an opponent's piece
        if to_pos in board and board[to_pos][0] == opponent:
            captured_piece = board[to_pos]
            score += self.weights['piece_count']
            
            # Huge bonus for capturing opponent's master
            if captured_piece[1] == MASTER:
                score += self.weights['master_capture']
        
        # 2. Master safety
        # If the moving piece is a master, evaluate its new position safety
        if is_master:
            # Check if we're moving to opponent's shrine (instant win)
            enemy_shrine = RED_SHRINE if self.player_id == BLUE else BLUE_SHRINE
            if to_pos == enemy_shrine:
                score += self.weights['master_to_shrine']
            
            # Evaluate how exposed the master is in the new position
            # For simplicity, we'll just check if it's on the edge
            to_x, to_y = to_pos
            edge_safety = 0
            if to_x in (0, BOARD_SIZE-1) or to_y in (0, BOARD_SIZE-1):
                edge_safety = 1  # Master is safer on the edge
            score += edge_safety * self.weights['master_safety']
        
        # 3. Center control
        # Pieces closer to the center have more mobility
        to_x, to_y = to_pos
        center_x, center_y = BOARD_SIZE // 2, BOARD_SIZE // 2
        center_distance = abs(to_x - center_x) + abs(to_y - center_y)
        center_control = (BOARD_SIZE - center_distance) / BOARD_SIZE
        score += center_control * self.weights['center_control']
        
        # 4. Distance to opponent's shrine (for master)
        if is_master:
            enemy_shrine = RED_SHRINE if self.player_id == BLUE else BLUE_SHRINE
            shrine_x, shrine_y = enemy_shrine
            distance = abs(to_x - shrine_x) + abs(to_y - shrine_y)
            
            # Normalize distance and invert (closer is better)
            max_distance = BOARD_SIZE * 2
            shrine_proximity = (max_distance - distance) / max_distance
            score += shrine_proximity * self.weights['shrine_distance']
        
        # 5. Attack opportunity
        # After this move, how many opponent pieces can we potentially capture next turn?
        # This is complex to accurately calculate without simulating, so we'll use a simpler proxy:
        # Check if we're moving next to opponent pieces
        attack_opportunities = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                adjacent_pos = (to_x + dx, to_y + dy)
                if (0 <= adjacent_pos[0] < BOARD_SIZE and 
                    0 <= adjacent_pos[1] < BOARD_SIZE and
                    adjacent_pos in board and 
                    board[adjacent_pos][0] == opponent):
                    attack_opportunities += 1
        
        score += attack_opportunities * self.weights['attack_opportunity']
        
        # Add a small random factor to break ties (if randomization is enabled)
        if self.randomize:
            score += random.uniform(0, 0.1)
        
        return score
