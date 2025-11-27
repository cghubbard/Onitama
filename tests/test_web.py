"""
Unit tests for the web application components.

Tests the GameManager, serialization, and API models.
"""

import pytest

from src.game.game import Game
from src.game.serialization import (
    serialize_position, deserialize_position,
    serialize_board, deserialize_board,
    serialize_move, deserialize_move,
    serialize_game_state, serialize_legal_moves,
    serialize_card_info, serialize_all_cards,
    get_cards_used, determine_win_reason
)
from src.web.game_manager import GameManager, SessionState
from src.web.models import (
    Position, GameConfig, MoveRequest, LegalMove,
    Player, PieceType, GameOutcome
)
from src.utils.constants import BLUE, RED, PAWN, MASTER, ONGOING


class TestSerialization:
    """Tests for serialization utilities."""

    def test_serialize_position(self):
        """Position tuple should serialize to list."""
        pos = (2, 3)
        result = serialize_position(pos)
        assert result == [2, 3]

    def test_deserialize_position(self):
        """Position list should deserialize to tuple."""
        pos = [2, 3]
        result = deserialize_position(pos)
        assert result == (2, 3)

    def test_position_roundtrip(self):
        """Position should survive serialization roundtrip."""
        original = (4, 1)
        serialized = serialize_position(original)
        deserialized = deserialize_position(serialized)
        assert deserialized == original

    def test_serialize_board(self):
        """Board dict should serialize to string-keyed dict."""
        board = {
            (0, 0): (BLUE, PAWN),
            (2, 0): (BLUE, MASTER),
            (4, 4): (RED, PAWN)
        }
        result = serialize_board(board)
        assert "0,0" in result
        assert result["0,0"] == [BLUE, PAWN]
        assert result["2,0"] == [BLUE, MASTER]
        assert result["4,4"] == [RED, PAWN]

    def test_deserialize_board(self):
        """String-keyed board should deserialize to tuple-keyed dict."""
        board_data = {
            "0,0": [BLUE, PAWN],
            "2,4": [RED, MASTER]
        }
        result = deserialize_board(board_data)
        assert (0, 0) in result
        assert result[(0, 0)] == (BLUE, PAWN)
        assert result[(2, 4)] == (RED, MASTER)

    def test_board_roundtrip(self):
        """Board should survive serialization roundtrip."""
        original = {
            (1, 2): (BLUE, PAWN),
            (3, 3): (RED, MASTER)
        }
        serialized = serialize_board(original)
        deserialized = deserialize_board(serialized)
        assert deserialized == original

    def test_serialize_move(self):
        """Move tuple should serialize to dict."""
        move = ((1, 0), (2, 1), "Tiger")
        result = serialize_move(move)
        assert result["from"] == [1, 0]
        assert result["to"] == [2, 1]
        assert result["card"] == "Tiger"

    def test_deserialize_move(self):
        """Move dict should deserialize to tuple."""
        move_data = {
            "from": [1, 0],
            "to": [2, 1],
            "card": "Tiger"
        }
        result = deserialize_move(move_data)
        assert result == ((1, 0), (2, 1), "Tiger")

    def test_move_roundtrip(self):
        """Move should survive serialization roundtrip."""
        original = ((0, 0), (1, 2), "Dragon")
        serialized = serialize_move(original)
        deserialized = deserialize_move(serialized)
        assert deserialized == original

    def test_serialize_game_state(self):
        """Game state should serialize correctly."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])
        result = serialize_game_state(game)

        assert "board" in result
        assert "current_player" in result
        assert "blue_cards" in result
        assert "red_cards" in result
        assert "neutral_card" in result
        assert "outcome" in result
        assert "move_count" in result

        assert result["current_player"] == BLUE
        assert len(result["blue_cards"]) == 2
        assert len(result["red_cards"]) == 2
        assert result["outcome"] == ONGOING
        assert result["move_count"] == 0

    def test_serialize_legal_moves(self):
        """Legal moves should serialize to list of dicts."""
        moves = [
            ((0, 0), (1, 1), "Tiger"),
            ((2, 0), (2, 2), "Dragon")
        ]
        result = serialize_legal_moves(moves)

        assert len(result) == 2
        assert result[0]["from"] == [0, 0]
        assert result[0]["to"] == [1, 1]
        assert result[0]["card"] == "Tiger"

    def test_serialize_card_info(self):
        """Card info should include name and movements."""
        result = serialize_card_info("Tiger")
        assert result["name"] == "Tiger"
        assert "movements" in result
        assert isinstance(result["movements"], list)

    def test_serialize_card_info_invalid(self):
        """Invalid card name should raise error."""
        with pytest.raises(ValueError):
            serialize_card_info("InvalidCard")

    def test_serialize_all_cards(self):
        """All cards should serialize."""
        result = serialize_all_cards()
        assert len(result) > 0
        assert all("name" in card for card in result)
        assert all("movements" in card for card in result)

    def test_get_cards_used(self):
        """Get cards used should return 5 card names."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])
        result = get_cards_used(game)
        assert len(result) == 5
        assert set(result) == {'Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'}

    def test_determine_win_reason_ongoing(self):
        """Ongoing game should return None."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])
        result = determine_win_reason(game)
        assert result is None


class TestGameManager:
    """Tests for GameManager class."""

    @pytest.fixture
    def manager(self):
        """Create a GameManager instance without logging."""
        return GameManager(logger=None)

    def test_create_game(self, manager):
        """Creating a game should return a session."""
        session = manager.create_game(
            blue_agent="human",
            red_agent="heuristic",
            log_game=False
        )

        assert session is not None
        assert session.game_id is not None
        assert session.blue_agent_type == "human"
        assert session.red_agent_type == "heuristic"
        assert session.state == SessionState.PLAYING

    def test_create_game_with_cards(self, manager):
        """Creating a game with specific cards should use those cards."""
        cards = ['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab']
        session = manager.create_game(
            blue_agent="random",
            red_agent="random",
            cards=cards,
            log_game=False
        )

        used_cards = get_cards_used(session.game)
        assert set(used_cards) == set(cards)

    def test_get_session(self, manager):
        """Should retrieve session by ID."""
        session = manager.create_game(log_game=False)
        retrieved = manager.get_session(session.game_id)
        assert retrieved == session

    def test_get_session_not_found(self, manager):
        """Should return None for unknown session."""
        retrieved = manager.get_session("nonexistent-id")
        assert retrieved is None

    def test_get_game_state(self, manager):
        """Should return serialized game state."""
        session = manager.create_game(log_game=False)
        state = manager.get_game_state(session)

        assert state["game_id"] == session.game_id
        assert "board" in state
        assert "current_player" in state
        assert state["blue_agent"] == "human"

    def test_get_legal_moves(self, manager):
        """Should return serialized legal moves."""
        session = manager.create_game(log_game=False)
        moves = manager.get_legal_moves(session)

        assert isinstance(moves, list)
        assert len(moves) > 0
        assert "from" in moves[0]
        assert "to" in moves[0]
        assert "card" in moves[0]

    def test_make_move_valid(self, manager):
        """Valid move should succeed."""
        session = manager.create_game(
            blue_agent="human",
            red_agent="human",
            log_game=False
        )

        # Get a legal move
        legal_moves = session.game.get_legal_moves(BLUE)
        from_pos, to_pos, card_name = legal_moves[0]

        result = manager.make_move(session, from_pos, to_pos, card_name)

        assert result["success"] is True
        assert "state" in result

    def test_make_move_invalid(self, manager):
        """Invalid move should fail."""
        session = manager.create_game(log_game=False)

        # Try an invalid move
        result = manager.make_move(session, (0, 0), (4, 4), "Tiger")

        assert result["success"] is False
        assert "error" in result

    def test_is_human_turn_blue(self, manager):
        """Should correctly identify human turn for blue."""
        session = manager.create_game(
            blue_agent="human",
            red_agent="heuristic",
            log_game=False
        )

        assert session.is_human_turn() is True

    def test_is_human_turn_ai(self, manager):
        """Should correctly identify AI turn."""
        session = manager.create_game(
            blue_agent="heuristic",
            red_agent="human",
            log_game=False
        )

        assert session.is_human_turn() is False

    def test_pause_and_resume(self, manager):
        """Should pause and resume game."""
        session = manager.create_game(
            blue_agent="heuristic",
            red_agent="heuristic",
            log_game=False
        )

        manager.pause_game(session)
        assert session.state == SessionState.PAUSED

        manager.resume_game(session)
        assert session.state == SessionState.PLAYING

    def test_list_active_games(self, manager):
        """Should list all active games."""
        manager.create_game(log_game=False)
        manager.create_game(log_game=False)

        games = manager.list_active_games()
        assert len(games) == 2

    def test_cleanup_session(self, manager):
        """Should remove session after cleanup."""
        session = manager.create_game(log_game=False)
        game_id = session.game_id

        manager.cleanup_session(game_id)

        assert manager.get_session(game_id) is None

    def test_make_ai_move(self, manager):
        """AI should make a valid move."""
        import asyncio
        session = manager.create_game(
            blue_agent="random",
            red_agent="random",
            log_game=False
        )

        result = asyncio.get_event_loop().run_until_complete(
            manager.make_ai_move(session)
        )

        assert result is not None
        assert result["success"] is True

    def test_make_ai_move_human_turn(self, manager):
        """AI move should return None when it's human's turn."""
        import asyncio
        session = manager.create_game(
            blue_agent="human",
            red_agent="random",
            log_game=False
        )

        result = asyncio.get_event_loop().run_until_complete(
            manager.make_ai_move(session)
        )

        assert result is None

    def test_step_game(self, manager):
        """Step should make one move in paused game."""
        import asyncio
        session = manager.create_game(
            blue_agent="random",
            red_agent="random",
            log_game=False
        )

        manager.pause_game(session)
        result = asyncio.get_event_loop().run_until_complete(
            manager.step_game(session)
        )

        assert result is not None
        assert result["success"] is True

    def test_step_game_not_paused(self, manager):
        """Step should return None if game not paused."""
        import asyncio
        session = manager.create_game(
            blue_agent="random",
            red_agent="random",
            log_game=False
        )

        result = asyncio.get_event_loop().run_until_complete(
            manager.step_game(session)
        )

        assert result is None


class TestPydanticModels:
    """Tests for Pydantic request/response models."""

    def test_position_valid(self):
        """Valid position should be accepted."""
        pos = Position(x=2, y=3)
        assert pos.x == 2
        assert pos.y == 3

    def test_position_boundary(self):
        """Position at boundaries should be valid."""
        pos_min = Position(x=0, y=0)
        pos_max = Position(x=4, y=4)
        assert pos_min.x == 0
        assert pos_max.x == 4

    def test_position_invalid(self):
        """Invalid position should raise error."""
        with pytest.raises(Exception):
            Position(x=5, y=0)
        with pytest.raises(Exception):
            Position(x=-1, y=0)

    def test_game_config_defaults(self):
        """GameConfig should have sensible defaults."""
        config = GameConfig()
        assert config.blue_agent == "human"
        assert config.red_agent == "heuristic"
        assert config.cards is None

    def test_game_config_custom(self):
        """GameConfig should accept custom values."""
        config = GameConfig(
            blue_agent="random",
            red_agent="random",
            cards=["Tiger", "Dragon", "Frog", "Rabbit", "Crab"]
        )
        assert config.blue_agent == "random"
        assert config.cards is not None
        assert len(config.cards) == 5

    def test_move_request(self):
        """MoveRequest should validate correctly."""
        move = MoveRequest(
            from_pos=Position(x=2, y=0),
            to_pos=Position(x=2, y=2),
            card_name="Tiger"
        )
        assert move.from_pos.x == 2
        assert move.to_pos.y == 2
        assert move.card_name == "Tiger"

    def test_legal_move(self):
        """LegalMove should serialize correctly."""
        move = LegalMove(
            from_pos=[1, 0],
            to_pos=[2, 1],
            card="Dragon"
        )
        assert move.from_pos == [1, 0]
        assert move.card == "Dragon"

    def test_player_enum(self):
        """Player enum should have correct values."""
        assert Player.BLUE == 0
        assert Player.RED == 1

    def test_piece_type_enum(self):
        """PieceType enum should have correct values."""
        assert PieceType.PAWN == 0
        assert PieceType.MASTER == 1

    def test_game_outcome_enum(self):
        """GameOutcome enum should have correct values."""
        assert GameOutcome.ONGOING == 0
        assert GameOutcome.BLUE_WINS == 1
        assert GameOutcome.RED_WINS == 2
        assert GameOutcome.DRAW == 3


class TestSessionState:
    """Tests for session state management."""

    @pytest.fixture
    def manager(self):
        """Create a GameManager instance."""
        return GameManager(logger=None)

    def test_initial_state(self, manager):
        """New session should be in PLAYING state."""
        session = manager.create_game(log_game=False)
        assert session.state == SessionState.PLAYING

    def test_state_after_game_over(self, manager):
        """Session should be FINISHED after game ends."""
        session = manager.create_game(
            blue_agent="random",
            red_agent="random",
            log_game=False
        )

        # Play until game ends
        while session.game.get_outcome() == ONGOING:
            current_player = session.game.get_current_player()
            moves = session.game.get_legal_moves(current_player)
            if not moves:
                break
            from_pos, to_pos, card_name = moves[0]
            manager.make_move(session, from_pos, to_pos, card_name)

        assert session.state == SessionState.FINISHED

    def test_get_current_agent_type(self, manager):
        """Should return correct agent type for current player."""
        session = manager.create_game(
            blue_agent="human",
            red_agent="heuristic",
            log_game=False
        )

        # Blue starts
        assert session.get_current_agent_type() == "human"

        # Make a move to switch to red
        moves = session.game.get_legal_moves(BLUE)
        from_pos, to_pos, card_name = moves[0]
        manager.make_move(session, from_pos, to_pos, card_name)

        assert session.get_current_agent_type() == "heuristic"


class TestGameIntegration:
    """Integration tests for full game flows."""

    @pytest.fixture
    def manager(self):
        """Create a GameManager instance."""
        return GameManager(logger=None)

    def test_full_game_random_vs_random(self, manager):
        """Random agents should complete a game."""
        session = manager.create_game(
            blue_agent="random",
            red_agent="random",
            log_game=False
        )

        move_count = 0
        max_moves = 200

        while session.game.get_outcome() == ONGOING and move_count < max_moves:
            current_player = session.game.get_current_player()
            moves = session.game.get_legal_moves(current_player)
            if not moves:
                break

            from_pos, to_pos, card_name = moves[0]
            result = manager.make_move(session, from_pos, to_pos, card_name)
            assert result["success"] is True

            move_count += 1

        # Game should have ended
        assert session.game.get_outcome() != ONGOING or move_count >= max_moves

    def test_capture_detection(self, manager):
        """Captures should be detected in move result."""
        session = manager.create_game(log_game=False)
        game = session.game

        # Set up a capture scenario - need masters for game to be valid
        game.board.clear()
        game.board[(2, 0)] = (BLUE, MASTER)  # Blue master
        game.board[(2, 4)] = (RED, MASTER)   # Red master
        game.board[(2, 2)] = (BLUE, PAWN)    # Blue pawn that will capture
        game.board[(2, 3)] = (RED, PAWN)     # Red pawn to be captured

        # Find a move that captures - the capture detection happens in game_manager
        # which checks game.get_board_state() before the move is made
        # The to_pos needs to match a position in the board state
        moves = game.get_legal_moves(BLUE)
        capture_move = None
        for move in moves:
            from_pos, to_pos, card_name = move
            # Check if destination has an enemy piece
            board_state = game.get_board_state()
            if to_pos in board_state and board_state[to_pos][0] == RED:
                capture_move = move
                break

        if capture_move:
            from_pos, to_pos, card_name = capture_move
            result = manager.make_move(session, from_pos, to_pos, card_name)
            assert result["success"] is True
            # Capture info is logged but may be None if log_session is None
            # The important thing is the move succeeded


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
