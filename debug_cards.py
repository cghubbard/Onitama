"""
Debug script to visualize card movements in Onitama.
"""
import sys
from src.utils.debug_utils import visualize_card_moves, visualize_all_cards
from src.utils.constants import MOVE_CARDS

def main():
    """
    Main function to visualize card movements.
    If a card name is provided as a command-line argument, 
    visualize just that card. Otherwise, visualize all cards.
    """
    if len(sys.argv) > 1:
        card_name = sys.argv[1]
        if card_name in MOVE_CARDS:
            visualize_card_moves(card_name)
        else:
            print(f"Error: Card '{card_name}' not found!")
            print(f"Available cards: {', '.join(sorted(MOVE_CARDS.keys()))}")
    else:
        # No specific card mentioned, show all cards
        visualize_all_cards()

if __name__ == "__main__":
    main()
