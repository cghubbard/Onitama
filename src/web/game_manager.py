"""
Game session manager for the web interface.

Manages active game sessions, handling both human vs AI and AI vs AI games.
"""
import uuid
import asyncio
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from src.game.game import Game
from src.game.serialization import (
    serialize_game_state, serialize_legal_moves, serialize_move,
    get_cards_used, determine_win_reason
)
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.logging.game_logger import GameLogger, LogMode
from src.utils.constants import BLUE, RED, ONGOING


class SessionState(Enum):
    """State of a game session."""
    WAITING = "waiting"      # Waiting for players
    PLAYING = "playing"      # Game in progress
    PAUSED = "paused"        # AI game paused
    FINISHED = "finished"    # Game over


@dataclass
class GameSession:
    """An active game session."""
    game_id: str
    game: Game
    blue_agent_type: str
    red_agent_type: str
    state: SessionState = SessionState.WAITING
    log_session: Optional[Any] = None
    ai_task: Optional[asyncio.Task] = None
    ai_speed: float = 1.0  # Seconds between AI moves
    observers: List[Callable] = field(default_factory=list)

    def is_human_turn(self) -> bool:
        """Check if it's a human player's turn."""
        current = self.game.get_current_player()
        agent_type = self.blue_agent_type if current == BLUE else self.red_agent_type
        return agent_type == "human"

    def get_current_agent_type(self) -> str:
        """Get the agent type for the current player."""
        current = self.game.get_current_player()
        return self.blue_agent_type if current == BLUE else self.red_agent_type


class GameManager:
    """
    Manages active game sessions.

    Handles game creation, move execution, AI moves, and session cleanup.
    """

    def __init__(self, logger: Optional[GameLogger] = None, data_dir: str = "data"):
        """
        Initialize game manager.

        Args:
            logger: Optional GameLogger for logging games
            data_dir: Directory for data storage
        """
        self.sessions: Dict[str, GameSession] = {}
        self.logger = logger or GameLogger(mode=LogMode.ALL, data_dir=data_dir)
        self._agents_cache: Dict[str, Any] = {}

    def _create_agent(self, agent_type: str, player: int):
        """Create an agent instance."""
        if agent_type == "random":
            return RandomAgent(player)
        elif agent_type == "heuristic":
            return HeuristicAgent(player)
        elif agent_type == "human":
            return None
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def create_game(
        self,
        blue_agent: str = "human",
        red_agent: str = "heuristic",
        cards: Optional[List[str]] = None,
        log_game: bool = True
    ) -> GameSession:
        """
        Create a new game session.

        Args:
            blue_agent: Agent type for blue player
            red_agent: Agent type for red player
            cards: Optional list of 5 card names
            log_game: Whether to log this game

        Returns:
            New GameSession
        """
        game_id = str(uuid.uuid4())
        game = Game(cards=cards)

        # Start logging session
        log_session = None
        if log_game:
            log_session = self.logger.start_game(game, blue_agent, red_agent)

        session = GameSession(
            game_id=game_id,
            game=game,
            blue_agent_type=blue_agent,
            red_agent_type=red_agent,
            state=SessionState.PLAYING,
            log_session=log_session
        )

        self.sessions[game_id] = session
        return session

    def get_session(self, game_id: str) -> Optional[GameSession]:
        """Get a game session by ID."""
        return self.sessions.get(game_id)

    def get_game_state(self, session: GameSession) -> Dict[str, Any]:
        """Get the current game state as a dictionary."""
        state = serialize_game_state(session.game)
        state["game_id"] = session.game_id
        state["blue_agent"] = session.blue_agent_type
        state["red_agent"] = session.red_agent_type
        return state

    def get_legal_moves(self, session: GameSession) -> List[Dict[str, Any]]:
        """Get legal moves for the current player."""
        moves = session.game.get_legal_moves(session.game.get_current_player())
        return serialize_legal_moves(moves)

    def make_move(
        self,
        session: GameSession,
        from_pos: tuple,
        to_pos: tuple,
        card_name: str
    ) -> Dict[str, Any]:
        """
        Make a move in a game.

        Args:
            session: Game session
            from_pos: (x, y) position to move from
            to_pos: (x, y) position to move to
            card_name: Name of card to use

        Returns:
            Dict with move result
        """
        game = session.game

        # Check if game is still ongoing
        if game.get_outcome() != ONGOING:
            return {"success": False, "error": "Game is already over"}

        # Log pre-move state
        pre_state = None
        capture = None
        if session.log_session:
            pre_state = session.log_session.log_pre_move_state(game)
            # Check for capture
            if to_pos in game.get_board_state():
                capture = game.get_board_state()[to_pos]

        # Make the move
        move = (from_pos, to_pos, card_name)
        success = game.make_move(from_pos, to_pos, card_name)

        if not success:
            return {"success": False, "error": "Invalid move"}

        # Log the move
        if session.log_session and pre_state:
            session.log_session.log_move_with_pre_state(pre_state, move)

        # Check for game over
        outcome = game.get_outcome()
        result = {
            "success": True,
            "state": self.get_game_state(session),
            "game_over": outcome != ONGOING,
            "capture": list(capture) if capture else None
        }

        if outcome != ONGOING:
            session.state = SessionState.FINISHED
            winner = None
            if outcome == 1:  # BLUE_WINS
                winner = BLUE
            elif outcome == 2:  # RED_WINS
                winner = RED

            result["winner"] = winner
            result["win_reason"] = determine_win_reason(game)

            # End logging
            if session.log_session:
                session.log_session.end_game(winner, result["win_reason"])

        return result

    async def make_ai_move(self, session: GameSession) -> Optional[Dict[str, Any]]:
        """
        Have the AI make a move.

        Args:
            session: Game session

        Returns:
            Move result or None if not AI's turn
        """
        game = session.game

        if game.get_outcome() != ONGOING:
            return None

        current_player = game.get_current_player()
        agent_type = session.get_current_agent_type()

        if agent_type == "human":
            return None

        # Create agent and get move
        agent = self._create_agent(agent_type, current_player)
        if agent is None:
            return None

        move = agent.select_move(game)
        if move is None:
            return {"success": False, "error": "No legal moves"}

        from_pos, to_pos, card_name = move
        return self.make_move(session, from_pos, to_pos, card_name)

    async def run_ai_game(
        self,
        session: GameSession,
        speed: float = 1.0,
        on_move: Optional[Callable] = None
    ):
        """
        Run an AI vs AI game.

        Args:
            session: Game session
            speed: Seconds between moves
            on_move: Callback after each move
        """
        session.ai_speed = speed

        while session.game.get_outcome() == ONGOING:
            if session.state == SessionState.PAUSED:
                await asyncio.sleep(0.1)
                continue

            if session.state != SessionState.PLAYING:
                break

            result = await self.make_ai_move(session)

            if on_move and result:
                await on_move(result)

            if result and result.get("game_over"):
                break

            await asyncio.sleep(speed)

    def pause_game(self, session: GameSession):
        """Pause an AI game."""
        if session.state == SessionState.PLAYING:
            session.state = SessionState.PAUSED

    def resume_game(self, session: GameSession):
        """Resume a paused AI game."""
        if session.state == SessionState.PAUSED:
            session.state = SessionState.PLAYING

    async def step_game(self, session: GameSession) -> Optional[Dict[str, Any]]:
        """Make one AI move in a paused game."""
        if session.state != SessionState.PAUSED:
            return None
        return await self.make_ai_move(session)

    def end_game(self, game_id: str, reason: str = "abandoned"):
        """End a game session."""
        session = self.sessions.get(game_id)
        if session:
            session.state = SessionState.FINISHED

            # Cancel any running AI task
            if session.ai_task and not session.ai_task.done():
                session.ai_task.cancel()

            # End logging if game wasn't finished normally
            if session.log_session and session.game.get_outcome() == ONGOING:
                session.log_session.end_game(None, reason)

    def cleanup_session(self, game_id: str):
        """Remove a session from memory."""
        if game_id in self.sessions:
            self.end_game(game_id, "cleanup")
            del self.sessions[game_id]

    def list_active_games(self) -> List[Dict[str, Any]]:
        """List all active game sessions."""
        return [
            {
                "game_id": session.game_id,
                "blue_agent": session.blue_agent_type,
                "red_agent": session.red_agent_type,
                "state": session.state.value,
                "move_count": len(session.game.move_history),
                "current_player": session.game.get_current_player()
            }
            for session in self.sessions.values()
        ]
