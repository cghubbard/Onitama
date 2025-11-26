"""
Renderer for Onitama game.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.game import Game
from src.utils.constants import BOARD_SIZE, BLUE, RED, PAWN, MASTER


class ConsoleRenderer:
    """
    Renders the Onitama game in the console.
    
    This provides a simple text-based visualization of the game state.
    """
    
    @staticmethod
    def render(game: 'Game'):
        """
        Render the game to the console.
        
        Args:
            game: The Game object to render
        """
        board = game.get_board_state()
        print(game)  # Uses the game's __str__ method


class ASCIIRenderer:
    """
    Renders the Onitama game with rich ASCII art.
    
    This provides a more visually appealing text-based visualization.
    """
    
    @staticmethod
    def render(game: 'Game'):
        """
        Render the game with ASCII art.
        
        Args:
            game: The Game object to render
        """
        board = game.get_board_state()
        current_player = game.get_current_player()
        outcome = game.get_outcome()
        blue_cards = game.get_player_cards(BLUE)
        red_cards = game.get_player_cards(RED)
        neutral_card = game.get_neutral_card()
        
        # Horizontal line
        h_line = "+---" * BOARD_SIZE + "+"
        
        # Print board with coordinates (using x,y coordinates)
        print("  " + " ".join(str(i) for i in range(BOARD_SIZE)))
        print(" " + h_line)
        
        for y in range(BOARD_SIZE):
            row = f"{y}|"
            for x in range(BOARD_SIZE):
                pos = (x, y)  # Using (x,y) coordinates
                if pos in board:
                    player, piece_type = board[pos]
                    if player == BLUE:
                        symbol = "B" if piece_type == MASTER else "b"
                    else:  # RED
                        symbol = "R" if piece_type == MASTER else "r"
                else:
                    # Mark shrines with special symbols
                    if pos == (2, 0):  # Blue shrine (x=2, y=0)
                        symbol = "^"
                    elif pos == (2, 4):  # Red shrine (x=2, y=4)
                        symbol = "v"
                    else:
                        symbol = " "
                row += f" {symbol} |"
            print(row)
            print(" " + h_line)
        
        # Print player info and cards
        from src.utils.constants import PLAYER_NAMES, OUTCOME_NAMES
        
        print(f"\nCurrent player: {PLAYER_NAMES[current_player]}")
        print(f"Game state: {OUTCOME_NAMES[outcome]}")
        
        print("\nBLUE cards:")
        for card in blue_cards:
            print(f"  {card.name}")
        
        print("\nRED cards:")
        for card in red_cards:
            print(f"  {card.name}")
        
        print(f"\nNeutral card: {neutral_card.name}")
