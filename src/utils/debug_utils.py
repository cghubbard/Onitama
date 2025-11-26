"""
Debug utilities for Onitama game.
"""
from src.utils.constants import MOVE_CARDS, BLUE, RED
from src.game.card import Card


def visualize_card_moves(card_name):
    """
    Visualize possible moves for a card from both BLUE and RED perspectives.
    
    Args:
        card_name: Name of the card to visualize
    
    Prints:
        ASCII visualization of possible moves for both players
    """
    if card_name not in MOVE_CARDS:
        print(f"Card '{card_name}' not found!")
        return
    
    # Create the card
    card = Card(card_name, MOVE_CARDS[card_name])
    
    # Get movements for both players
    blue_movements = card.get_movements(BLUE)
    red_movements = card.get_movements(RED)
    
    # Create boards for visualization
    # Use a 5x5 grid with the piece in the middle (position 2,2)
    blue_board = [[' ' for _ in range(5)] for _ in range(5)]
    red_board = [[' ' for _ in range(5)] for _ in range(5)]
    
    # Mark the piece position
    blue_board[2][2] = 'B'
    red_board[2][2] = 'R'
    
    # Mark landing spots for BLUE
    for dx, dy in blue_movements:
        # Calculate landing position (from center)
        # Note: In our board visualization, x is horizontal (column) and y is vertical (row)
        # So dx affects column (x) and dy affects row (y)
        x = 2 + dx  # Center column (2) + horizontal movement
        y = 2 + dy  # Center row (2) + vertical movement
        # Check if position is on the board
        if 0 <= x < 5 and 0 <= y < 5:
            blue_board[y][x] = 'X'
    
    # Mark landing spots for RED
    # In this visualization, we're showing RED's moves from the same board orientation as BLUE
    # so we need to apply the actual RED movements without re-mirroring
    for dx, dy in red_movements:
        # Calculate landing position (from center) 
        x = 2 + dx  # Column
        y = 2 + dy  # Row
        # Check if position is on the board
        if 0 <= x < 5 and 0 <= y < 5:
            red_board[y][x] = 'X'
    
    # Print card name
    print(f"\n=== {card_name} Card Movements ===\n")
    
    # Print BLUE perspective
    print("BLUE Movements (Board viewed from RED's perspective):")
    print(red_movements)
    print("Blue at top (y=0), Red at bottom (y=4)")
    print("  0 1 2 3 4  (x coordinates)")
    print(" +---------+")
    for i, row in enumerate(blue_board):
        print(f"{i}|{' '.join(row)}|  (y={i})")
    print(" +---------+")
    
    # Print RED perspective
    print("\nRED Movements (Board viewed from RED's perspective):")
    print(red_movements)
    print("Blue at top (y=0), Red at bottom (y=4)")
    print("  0 1 2 3 4  (x coordinates)")
    print(" +---------+")
    for i, row in enumerate(red_board):
        print(f"{i}|{' '.join(row)}|  (y={i})")
    print(" +---------+")


def visualize_all_cards():
    """
    Visualize movements for all available cards.
    """
    for card_name in sorted(MOVE_CARDS.keys()):
        visualize_card_moves(card_name)
        print("\n" + "="*40 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Visualizing all cards:")
    visualize_all_cards()
    
    # Or visualize a specific card
    # visualize_card_moves("Tiger")
