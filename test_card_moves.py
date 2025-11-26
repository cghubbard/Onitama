"""
Test script to verify card moves for both players.
"""
from src.utils.constants import MOVE_CARDS
from src.game.card import Card

def test_card_moves():
    """Test card moves for both BLUE and RED players."""
    # Test a selection of cards with different movement patterns
    test_cards = ['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Monkey', 'Elephant']
    
    for name in test_cards:
        card = Card(name, MOVE_CARDS[name])
        print(f'Card: {name}')
        print(f'Original moves: {MOVE_CARDS[name]}')
        print(f'BLUE moves: {card.get_movements(0)}')
        print(f'RED moves: {card.get_movements(1)}')
        print()

if __name__ == "__main__":
    test_card_moves()
