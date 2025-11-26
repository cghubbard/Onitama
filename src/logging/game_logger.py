"""
Game logger for capturing game trajectories.

Provides an easy interface to log games during play, with support for
configurable logging modes (none, all, sample).
"""
import random
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum

from src.logging.trajectory import (
    GameTrajectory, GameConfig, Transition, StateSnapshot
)
from src.logging.storage import GameStorage
from src.game.serialization import (
    serialize_board, serialize_legal_moves, serialize_move
)
from src.utils.constants import BLUE, RED, ONGOING


class LogMode(Enum):
    """Logging mode options."""
    NONE = "none"      # Don't log any games
    ALL = "all"        # Log all games
    SAMPLE = "sample"  # Log with sampling rate


class GameLogger:
    """
    Logs game trajectories for replay and RL training.

    Usage:
        logger = GameLogger(storage, mode=LogMode.ALL)

        # Start a new game
        session = logger.start_game(game, "human", "heuristic")

        # Before each move, log the transition
        session.log_move(game, move, capture_info)

        # When game ends
        session.end_game(winner, reason)
    """

    def __init__(
        self,
        storage: Optional[GameStorage] = None,
        mode: LogMode = LogMode.ALL,
        sample_rate: float = 1.0,
        data_dir: str = "data"
    ):
        """
        Initialize game logger.

        Args:
            storage: GameStorage instance (created if None)
            mode: Logging mode
            sample_rate: Probability of logging each game (for SAMPLE mode)
            data_dir: Data directory for storage
        """
        self.storage = storage or GameStorage(data_dir)
        self.mode = mode
        self.sample_rate = sample_rate

    def should_log(self) -> bool:
        """Determine if current game should be logged based on mode."""
        if self.mode == LogMode.NONE:
            return False
        if self.mode == LogMode.ALL:
            return True
        # SAMPLE mode
        return random.random() < self.sample_rate

    def start_game(
        self,
        game: 'Game',
        blue_agent: str,
        red_agent: str
    ) -> 'GameLogSession':
        """
        Start logging a new game.

        Args:
            game: Game instance
            blue_agent: Type of blue agent ("human", "random", "heuristic", "ppo")
            red_agent: Type of red agent

        Returns:
            GameLogSession for logging moves
        """
        should_log = self.should_log()

        return GameLogSession(
            logger=self,
            game=game,
            blue_agent=blue_agent,
            red_agent=red_agent,
            active=should_log
        )


class GameLogSession:
    """
    Active logging session for a single game.

    Captures state snapshots and transitions as the game progresses.
    """

    def __init__(
        self,
        logger: GameLogger,
        game: 'Game',
        blue_agent: str,
        red_agent: str,
        active: bool = True
    ):
        """
        Initialize logging session.

        Args:
            logger: Parent GameLogger
            game: Game instance being logged
            blue_agent: Type of blue agent
            red_agent: Type of red agent
            active: Whether this session is actually logging
        """
        self.logger = logger
        self.active = active

        if not active:
            self.trajectory = None
            return

        # Get cards used in game
        from src.game.serialization import get_cards_used
        cards = get_cards_used(game)

        # Create trajectory
        self.trajectory = GameTrajectory(
            game_id="",  # Will be auto-generated
            timestamp="",  # Will be auto-generated
            config=GameConfig(
                cards_used=cards,
                blue_agent=blue_agent,
                red_agent=red_agent
            )
        )

    def capture_state(self, game: 'Game') -> StateSnapshot:
        """
        Capture current game state as a snapshot.

        Args:
            game: Game instance

        Returns:
            StateSnapshot of current state
        """
        return StateSnapshot(
            board=serialize_board(game.get_board_state()),
            current_player=game.get_current_player(),
            blue_cards=[c.name for c in game.get_player_cards(BLUE)],
            red_cards=[c.name for c in game.get_player_cards(RED)],
            neutral_card=game.get_neutral_card().name,
            move_number=len(game.move_history),
            outcome=game.get_outcome()
        )

    def log_move(
        self,
        game_before: 'Game',
        move: Tuple[Tuple[int, int], Tuple[int, int], str],
        capture: Optional[Tuple[int, int]] = None
    ):
        """
        Log a move transition.

        Call this BEFORE making the move, passing the game state
        and the move that will be made.

        Args:
            game_before: Game state before the move
            move: The move being made (from_pos, to_pos, card_name)
            capture: Captured piece info [player, piece_type] or None
        """
        if not self.active:
            return

        # Capture state before move
        state = self.capture_state(game_before)

        # Get legal moves
        legal_moves = serialize_legal_moves(
            game_before.get_legal_moves(game_before.get_current_player())
        )

        # Create transition
        transition = Transition(
            move_number=state.move_number,
            state=state,
            legal_moves=legal_moves,
            action=serialize_move(move),
            capture=list(capture) if capture else None
        )

        self.trajectory.add_transition(transition)

    def log_pre_move_state(self, game: 'Game') -> Dict[str, Any]:
        """
        Capture pre-move state for later use.

        Returns info needed to detect captures after move is made.

        Args:
            game: Game before move

        Returns:
            Dict with board state and legal moves
        """
        return {
            "board": game.get_board_state().copy(),
            "legal_moves": game.get_legal_moves(game.get_current_player()),
            "state_snapshot": self.capture_state(game) if self.active else None
        }

    def log_move_with_pre_state(
        self,
        pre_state: Dict[str, Any],
        move: Tuple[Tuple[int, int], Tuple[int, int], str]
    ):
        """
        Log a move using pre-captured state.

        Args:
            pre_state: Dict from log_pre_move_state()
            move: The move that was made
        """
        if not self.active:
            return

        from_pos, to_pos, card_name = move

        # Check if there was a capture
        capture = None
        if to_pos in pre_state["board"]:
            piece = pre_state["board"][to_pos]
            capture = list(piece)

        # Create transition
        transition = Transition(
            move_number=pre_state["state_snapshot"].move_number,
            state=pre_state["state_snapshot"],
            legal_moves=serialize_legal_moves(pre_state["legal_moves"]),
            action=serialize_move(move),
            capture=capture
        )

        self.trajectory.add_transition(transition)

    def end_game(self, winner: Optional[int], reason: str) -> Optional[str]:
        """
        End the game and save trajectory.

        Args:
            winner: Winner (0=BLUE, 1=RED, None for draw)
            reason: Win reason ("master_captured", "shrine_reached", "draw", "max_moves")

        Returns:
            game_id if saved, None if not logging
        """
        if not self.active:
            return None

        self.trajectory.set_outcome(winner, reason)

        # Save to storage
        game_id = self.logger.storage.save_trajectory(self.trajectory)

        return game_id

    @property
    def game_id(self) -> Optional[str]:
        """Get the game ID of this session."""
        return self.trajectory.game_id if self.trajectory else None


def create_logger_from_args(
    log_mode: str,
    sample_rate: float = 0.1,
    data_dir: str = "data"
) -> GameLogger:
    """
    Create a GameLogger from command-line arguments.

    Args:
        log_mode: "none", "all", or "sample"
        sample_rate: Sampling rate for "sample" mode
        data_dir: Data directory

    Returns:
        Configured GameLogger
    """
    mode_map = {
        "none": LogMode.NONE,
        "all": LogMode.ALL,
        "sample": LogMode.SAMPLE
    }

    mode = mode_map.get(log_mode.lower(), LogMode.NONE)

    return GameLogger(
        mode=mode,
        sample_rate=sample_rate,
        data_dir=data_dir
    )
