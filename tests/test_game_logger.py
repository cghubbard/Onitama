"""
Tests for game logging, especially terminal state logging.
"""
import pytest
import tempfile
import shutil
from src.game.game import Game
from src.logging.game_logger import GameLogger, LogMode
from src.logging.storage import GameStorage
from src.utils.constants import BLUE, RED, BLUE_WINS, RED_WINS, ONGOING


class TestTerminalStateLogging:
    """Tests for terminal state logging in game_logger.py"""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary game storage for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = GameStorage(data_dir=temp_dir)
        yield storage
        shutil.rmtree(temp_dir)

    def test_log_terminal_state_blue_wins(self, temp_storage):
        """Terminal state logged correctly when blue wins."""
        # Create a game
        game = Game()

        # Set up a winning state for blue (simulate master capture)
        # We need to manipulate the game to be in a terminal state
        game.outcome = BLUE_WINS

        # Create logger and session
        logger = GameLogger(temp_storage, mode=LogMode.ALL)
        session = logger.start_game(game, "test_agent", "test_agent")

        # Log the terminal state
        session.log_terminal_state(game)

        # End game
        session.end_game(BLUE, "master_captured")

        # Load the trajectory
        trajectory = temp_storage.load_trajectory(session.game_id)

        # Should have one transition (the terminal state)
        assert len(trajectory.transitions) == 1

        # The transition should have the terminal state
        terminal_transition = trajectory.transitions[0]
        assert terminal_transition.state.outcome == BLUE_WINS
        assert terminal_transition.action is None  # Terminal state has no action
        assert terminal_transition.legal_moves == []

    def test_log_terminal_state_red_wins(self, temp_storage):
        """Terminal state logged correctly when red wins."""
        game = Game()
        game.outcome = RED_WINS

        logger = GameLogger(temp_storage, mode=LogMode.ALL)
        session = logger.start_game(game, "test_agent", "test_agent")

        session.log_terminal_state(game)
        session.end_game(RED, "master_captured")

        trajectory = temp_storage.load_trajectory(session.game_id)

        assert len(trajectory.transitions) == 1
        terminal_transition = trajectory.transitions[0]
        assert terminal_transition.state.outcome == RED_WINS
        assert terminal_transition.action is None

    def test_terminal_transition_has_no_action(self, temp_storage):
        """Terminal transitions have action=None."""
        game = Game()
        game.outcome = BLUE_WINS

        logger = GameLogger(temp_storage, mode=LogMode.ALL)
        session = logger.start_game(game, "test_agent", "test_agent")

        session.log_terminal_state(game)
        session.end_game(BLUE, "shrine_reached")

        trajectory = temp_storage.load_trajectory(session.game_id)
        terminal_transition = trajectory.transitions[0]

        # Terminal state should have no action (absorbing state)
        assert terminal_transition.action is None

    def test_terminal_transition_outcome_set(self, temp_storage):
        """Terminal state snapshot has outcome != 0."""
        game = Game()
        game.outcome = BLUE_WINS

        logger = GameLogger(temp_storage, mode=LogMode.ALL)
        session = logger.start_game(game, "test_agent", "test_agent")

        session.log_terminal_state(game)
        session.end_game(BLUE, "master_captured")

        trajectory = temp_storage.load_trajectory(session.game_id)
        terminal_snapshot = trajectory.transitions[0].state

        # Outcome should be set (not ONGOING)
        assert terminal_snapshot.outcome != ONGOING
        assert terminal_snapshot.outcome == BLUE_WINS

    def test_full_game_with_terminal(self, temp_storage):
        """Test logging a complete game with both regular and terminal transitions."""
        game = Game()

        logger = GameLogger(temp_storage, mode=LogMode.ALL)
        session = logger.start_game(game, "test_agent", "test_agent")

        # Log a couple of regular moves
        for _ in range(3):
            pre_state = session.log_pre_move_state(game)
            # Make a move
            moves = game.get_legal_moves(game.get_current_player())
            if moves:
                from_pos, to_pos, card_name = moves[0]
                game.make_move(from_pos, to_pos, card_name)
                session.log_move_with_pre_state(pre_state, (from_pos, to_pos, card_name))

                # Check if game is over
                if game.get_outcome() != ONGOING:
                    break

        # If game ended, log terminal state
        outcome = game.get_outcome()
        if outcome != ONGOING:
            session.log_terminal_state(game)
            winner = BLUE if outcome == BLUE_WINS else RED if outcome == RED_WINS else None
            session.end_game(winner, "test_reason")

            # Verify the trajectory
            trajectory = temp_storage.load_trajectory(session.game_id)

            # Last transition should be terminal
            last_transition = trajectory.transitions[-1]
            assert last_transition.state.outcome != ONGOING
            assert last_transition.action is None

            # All other transitions should have actions
            for trans in trajectory.transitions[:-1]:
                assert trans.action is not None
