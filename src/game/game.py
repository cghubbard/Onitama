"""
Game class for Onitama.
"""
import random
from typing import List, Tuple, Dict, Optional, Set

from src.utils.constants import (
    BOARD_SIZE, BLUE, RED, PLAYER_NAMES, 
    PAWN, MASTER, BLUE_SHRINE, RED_SHRINE,
    ONGOING, BLUE_WINS, RED_WINS, MOVE_CARDS
)
from src.game.card import Card


class Game:
    """
    Represents the Onitama game state and rules.
    
    Handles board state, move validation, and win conditions.
    """
    
    def __init__(self, cards: Optional[List[str]] = None):
        """
        Initialize a new game of Onitama.
        
        Args:
            cards: Optional list of 5 card names to use. If None, 5 random cards will be selected.
        """
        # Initialize empty board
        self.board = {}
        self.reset_board()
        
        # Select and distribute cards
        self.setup_cards(cards)
        
        # Blue goes first
        self.current_player = BLUE
        
        # Game state
        self.outcome = ONGOING
        self.move_history = []
    
    def reset_board(self):
        """Set up the initial board state with pieces in their starting positions."""
        self.board = {}
        
        # Set up BLUE pieces (top row) - using (x, y) coordinates
        for x in range(BOARD_SIZE):
            # Master in the middle, pawns elsewhere
            piece_type = MASTER if x == 2 else PAWN
            self.board[(x, 0)] = (BLUE, piece_type)
        
        # Set up RED pieces (bottom row) - using (x, y) coordinates
        for x in range(BOARD_SIZE):
            # Master in the middle, pawns elsewhere
            piece_type = MASTER if x == 2 else PAWN
            self.board[(x, 4)] = (RED, piece_type)
    
    def setup_cards(self, card_names: Optional[List[str]] = None):
        """
        Set up the five cards for the game.
        
        Args:
            card_names: Optional list of 5 card names to use. If None, 5 random cards are selected.
        """
        if card_names is None:
            # Randomly select 5 cards
            all_cards = list(MOVE_CARDS.keys())
            card_names = random.sample(all_cards, 5)
        else:
            # Validate provided card names
            if len(card_names) != 5 or not all(name in MOVE_CARDS for name in card_names):
                raise ValueError("Must provide exactly 5 valid card names")
        
        # Create Card objects
        cards = [Card(name, MOVE_CARDS[name]) for name in card_names]
        
        # Distribute cards: 2 for BLUE, 2 for RED, 1 neutral
        self.player_cards = {
            BLUE: [cards[0], cards[1]],
            RED: [cards[2], cards[3]]
        }
        self.neutral_card = cards[4]
        
        # Decide first player based on the neutral card (in real Onitama, the color
        # indicator on the neutral card determines who goes first)
        # Here we just stick with Blue going first
    
    def get_legal_moves(self, player: int) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """
        Get all legal moves for the given player.
        
        Args:
            player: Player to get moves for (BLUE or RED)
            
        Returns:
            List of (from_pos, to_pos, card_name) tuples representing legal moves
        """
        if self.outcome != ONGOING or player != self.current_player:
            return []
        
        legal_moves = []
        
        # For each card the player has
        for card in self.player_cards[player]:
            # Get the possible movement patterns for this player's perspective
            movements = card.get_movements(player)
            
            # For each piece the player has on the board
            for pos, (piece_player, _) in self.board.items():
                if piece_player != player:
                    continue
                
                # Try each movement pattern
                for dx, dy in movements:
                    # Apply the movement to the position
                    # Note: positions are now (x, y) where x is column and y is row
                    to_x, to_y = pos[0] + dx, pos[1] + dy
                    to_pos = (to_x, to_y)
                    
                    # Check if the move is valid
                    if self.is_valid_move(player, pos, to_pos):
                        legal_moves.append((pos, to_pos, card.name))
        
        return legal_moves
    
    def is_valid_move(self, player: int, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """
        Check if a move is valid.
        
        Args:
            player: Player making the move
            from_pos: Position to move from
            to_pos: Position to move to
            
        Returns:
            True if the move is valid, False otherwise
        """
        # Check if from_pos has player's piece
        if from_pos not in self.board or self.board[from_pos][0] != player:
            return False
        
        # Check if to_pos is on the board
        to_x, to_y = to_pos
        if not (0 <= to_x < BOARD_SIZE and 0 <= to_y < BOARD_SIZE):
            return False
        
        # Check if to_pos doesn't have player's own piece
        if to_pos in self.board and self.board[to_pos][0] == player:
            return False
        
        return True
    
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], card_name: str) -> bool:
        """
        Make a move in the game.
        
        Args:
            from_pos: Position to move from
            to_pos: Position to move to
            card_name: Name of the card to use for the move
            
        Returns:
            True if move was successful, False if invalid
        """
        player = self.current_player
        
        # Find the card in player's hand
        card_idx = None
        for i, card in enumerate(self.player_cards[player]):
            if card.name == card_name:
                card_idx = i
                break
        
        if card_idx is None:
            return False  # Player doesn't have this card
        
        card = self.player_cards[player][card_idx]
        
        # Check if move is valid according to card
        valid_move = False
        for dx, dy in card.get_movements(player):
            # Apply the movement to the position - positions are (x, y)
            if (from_pos[0] + dx, from_pos[1] + dy) == to_pos:
                valid_move = True
                break
        
        if not valid_move or not self.is_valid_move(player, from_pos, to_pos):
            return False
        
        # Move the piece
        piece = self.board[from_pos]
        del self.board[from_pos]
        
        # Capture opponent piece if present
        if to_pos in self.board:
            captured_piece = self.board[to_pos]
        
        # Place the piece at the new position
        self.board[to_pos] = piece
        
        # Swap used card with neutral card
        self.player_cards[player][card_idx] = self.neutral_card
        self.neutral_card = card
        
        # Record move
        self.move_history.append((player, from_pos, to_pos, card_name))
        
        # Check for win condition
        self.check_win_condition(player, to_pos)
        
        # Switch player if game is still ongoing
        if self.outcome == ONGOING:
            self.current_player = RED if player == BLUE else BLUE
        
        return True
    
    def check_win_condition(self, player: int, last_move_pos: Tuple[int, int]):
        """
        Check if the game has been won.
        
        Args:
            player: Player who just moved
            last_move_pos: Position of the last move
        """
        # Check if the last move was a capture
        # We need to check if we *captured* an opponent's master
        # We'll check our move history to see if we captured a piece
        was_capture = len(self.move_history) > 0 and self.move_history[-1][2] == last_move_pos
        
        # Find any opponent master on the board
        opponent = RED if player == BLUE else BLUE
        opponent_master_exists = False
        
        for pos, (piece_player, piece_type) in self.board.items():
            if piece_player == opponent and piece_type == MASTER:
                opponent_master_exists = True
                break
        
        # Victory if the opponent's master was captured (doesn't exist on the board)
        if not opponent_master_exists:
            if player == BLUE:
                self.outcome = BLUE_WINS
            else:
                self.outcome = RED_WINS
            return
        
        # Victory by reaching opponent's shrine with master
        # Need to check if it's the player's master that reached the opponent's shrine
        enemy_shrine = RED_SHRINE if player == BLUE else BLUE_SHRINE
        if (last_move_pos == enemy_shrine and 
            last_move_pos in self.board and 
            self.board[last_move_pos][0] == player and  # Check it's player's piece
            self.board[last_move_pos][1] == MASTER):    # Check it's a master
            if player == BLUE:
                self.outcome = BLUE_WINS
            else:
                self.outcome = RED_WINS
            return
        
        # Check if any player can't move (stalemate)
        # This is a bit computationally expensive, so we might not want to check every move
        blue_has_moves = len(self.get_legal_moves(BLUE)) > 0 if self.current_player == BLUE else True
        red_has_moves = len(self.get_legal_moves(RED)) > 0 if self.current_player == RED else True
        
        if not blue_has_moves and not red_has_moves:
            self.outcome = DRAW
            return
    
    def get_board_state(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Get the current board state.
        
        Returns:
            Dictionary mapping positions to (player, piece_type) tuples
        """
        return self.board.copy()
    
    def get_player_cards(self, player: int) -> List[Card]:
        """
        Get the cards for a player.
        
        Args:
            player: Player to get cards for
            
        Returns:
            List of Card objects
        """
        return self.player_cards[player].copy()
    
    def get_neutral_card(self) -> Card:
        """
        Get the neutral card.
        
        Returns:
            The neutral Card object
        """
        return self.neutral_card
    
    def get_current_player(self) -> int:
        """
        Get the current player.
        
        Returns:
            Current player (BLUE or RED)
        """
        return self.current_player
    
    def get_outcome(self) -> int:
        """
        Get the game outcome.
        
        Returns:
            Game outcome (ONGOING, BLUE_WINS, RED_WINS, or DRAW)
        """
        return self.outcome
    
    def __str__(self) -> str:
        """Return string representation of the game state."""
        result = []
        
        # Add board representation using (x,y) coordinates
        result.append("Board:")
        for y in range(BOARD_SIZE):
            row = []
            for x in range(BOARD_SIZE):
                pos = (x, y)  # Using (x,y) coordinates
                if pos in self.board:
                    player, piece_type = self.board[pos]
                    if player == BLUE:
                        symbol = "B" if piece_type == MASTER else "b"
                    else:  # RED
                        symbol = "R" if piece_type == MASTER else "r"
                else:
                    symbol = "."
                row.append(symbol)
            result.append(" ".join(row))
        
        # Add current player
        result.append(f"\nCurrent player: {PLAYER_NAMES[self.current_player]}")
        
        # Add cards information
        result.append("\nBLUE cards:")
        for card in self.player_cards[BLUE]:
            result.append(f"  {card.name}")
        
        result.append("\nRED cards:")
        for card in self.player_cards[RED]:
            result.append(f"  {card.name}")
        
        result.append(f"\nNeutral card: {self.neutral_card.name}")
        
        # Add game outcome if not ongoing
        if self.outcome != ONGOING:
            from src.utils.constants import OUTCOME_NAMES
            result.append(f"\nGame outcome: {OUTCOME_NAMES[self.outcome]}")
        
        return "\n".join(result)
