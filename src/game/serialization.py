"""
Serialization utilities for Onitama game state.

Converts game state to/from JSON-serializable dictionaries for:
- Web API responses
- Game logging for RL training
- Replay storage
"""
from typing import Dict, List, Tuple, Any, Optional
from src.utils.constants import BLUE, RED, PAWN, MASTER, MOVE_CARDS


def serialize_position(pos: Tuple[int, int]) -> List[int]:
    """Convert position tuple to JSON-serializable list."""
    return [pos[0], pos[1]]


def deserialize_position(pos: List[int]) -> Tuple[int, int]:
    """Convert position list back to tuple."""
    return (pos[0], pos[1])


def serialize_board(board: Dict[Tuple[int, int], Tuple[int, int]]) -> Dict[str, List[int]]:
    """
    Serialize board state to JSON-compatible dictionary.

    Args:
        board: Dict mapping (x, y) -> (player_id, piece_type)

    Returns:
        Dict mapping "x,y" string -> [player_id, piece_type]
    """
    return {
        f"{pos[0]},{pos[1]}": [player, piece_type]
        for pos, (player, piece_type) in board.items()
    }


def deserialize_board(board_data: Dict[str, List[int]]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Deserialize board state from JSON dictionary.

    Args:
        board_data: Dict mapping "x,y" string -> [player_id, piece_type]

    Returns:
        Dict mapping (x, y) tuple -> (player_id, piece_type) tuple
    """
    result = {}
    for pos_str, piece_data in board_data.items():
        x, y = map(int, pos_str.split(','))
        result[(x, y)] = (piece_data[0], piece_data[1])
    return result


def serialize_move(move: Tuple[Tuple[int, int], Tuple[int, int], str]) -> Dict[str, Any]:
    """
    Serialize a move to JSON-compatible dictionary.

    Args:
        move: (from_pos, to_pos, card_name) tuple

    Returns:
        Dict with 'from', 'to', and 'card' keys
    """
    from_pos, to_pos, card_name = move
    return {
        "from": serialize_position(from_pos),
        "to": serialize_position(to_pos),
        "card": card_name
    }


def deserialize_move(move_data: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int], str]:
    """
    Deserialize a move from JSON dictionary.

    Args:
        move_data: Dict with 'from', 'to', and 'card' keys

    Returns:
        (from_pos, to_pos, card_name) tuple
    """
    return (
        deserialize_position(move_data["from"]),
        deserialize_position(move_data["to"]),
        move_data["card"]
    )


def serialize_game_state(game: 'Game') -> Dict[str, Any]:
    """
    Serialize complete game state to JSON-compatible dictionary.

    Args:
        game: Game instance

    Returns:
        Dict containing all game state information
    """
    return {
        "board": serialize_board(game.get_board_state()),
        "current_player": game.get_current_player(),
        "blue_cards": [card.name for card in game.get_player_cards(BLUE)],
        "red_cards": [card.name for card in game.get_player_cards(RED)],
        "neutral_card": game.get_neutral_card().name,
        "outcome": game.get_outcome(),
        "move_count": len(game.move_history)
    }


def serialize_legal_moves(moves: List[Tuple[Tuple[int, int], Tuple[int, int], str]]) -> List[Dict[str, Any]]:
    """
    Serialize list of legal moves.

    Args:
        moves: List of (from_pos, to_pos, card_name) tuples

    Returns:
        List of serialized move dictionaries
    """
    return [serialize_move(move) for move in moves]


def serialize_card_info(card_name: str) -> Dict[str, Any]:
    """
    Serialize card information including movement patterns.

    Args:
        card_name: Name of the card

    Returns:
        Dict with card name and movements
    """
    if card_name not in MOVE_CARDS:
        raise ValueError(f"Unknown card: {card_name}")

    return {
        "name": card_name,
        "movements": MOVE_CARDS[card_name]
    }


def serialize_all_cards() -> List[Dict[str, Any]]:
    """
    Serialize all available cards.

    Returns:
        List of card info dictionaries
    """
    return [serialize_card_info(name) for name in MOVE_CARDS.keys()]


def get_cards_used(game: 'Game') -> List[str]:
    """
    Get list of all card names used in the game.

    Args:
        game: Game instance

    Returns:
        List of 5 card names
    """
    cards = []
    cards.extend(card.name for card in game.get_player_cards(BLUE))
    cards.extend(card.name for card in game.get_player_cards(RED))
    cards.append(game.get_neutral_card().name)
    return cards


def determine_win_reason(game: 'Game') -> Optional[str]:
    """
    Determine how the game was won (if it's over).

    Args:
        game: Game instance

    Returns:
        'master_captured', 'shrine_reached', 'draw', or None if ongoing
    """
    from src.utils.constants import ONGOING, DRAW, BLUE_WINS, RED_WINS, BLUE_SHRINE, RED_SHRINE

    outcome = game.get_outcome()
    if outcome == ONGOING:
        return None
    if outcome == DRAW:
        return "draw"

    # Determine if win was by shrine or capture
    winner = BLUE if outcome == BLUE_WINS else RED
    opponent_shrine = RED_SHRINE if winner == BLUE else BLUE_SHRINE

    # Check if winner's master is on opponent's shrine
    board = game.get_board_state()
    if opponent_shrine in board:
        player, piece_type = board[opponent_shrine]
        if player == winner and piece_type == MASTER:
            return "shrine_reached"

    # Otherwise it was a capture
    return "master_captured"
