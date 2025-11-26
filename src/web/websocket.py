"""
WebSocket handlers for real-time game communication.
"""
import json
import asyncio
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect

from src.web.game_manager import GameManager, GameSession, SessionState
from src.utils.constants import ONGOING


class ConnectionManager:
    """Manages WebSocket connections for game sessions."""

    def __init__(self, game_manager: GameManager):
        self.game_manager = game_manager
        # Map game_id -> set of connected WebSockets
        self.connections: Dict[str, Set[WebSocket]] = {}
        # Map websocket -> game_id
        self.socket_games: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, game_id: str):
        """Connect a WebSocket to a game."""
        await websocket.accept()

        if game_id not in self.connections:
            self.connections[game_id] = set()

        self.connections[game_id].add(websocket)
        self.socket_games[websocket] = game_id

    def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket."""
        game_id = self.socket_games.get(websocket)
        if game_id and game_id in self.connections:
            self.connections[game_id].discard(websocket)
            if not self.connections[game_id]:
                del self.connections[game_id]
        if websocket in self.socket_games:
            del self.socket_games[websocket]

    async def broadcast(self, game_id: str, message: dict):
        """Broadcast a message to all connections for a game."""
        if game_id in self.connections:
            dead_sockets = set()
            for websocket in self.connections[game_id]:
                try:
                    await websocket.send_json(message)
                except Exception:
                    dead_sockets.add(websocket)

            # Clean up dead connections
            for ws in dead_sockets:
                self.disconnect(ws)

    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send a message to a specific WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)


class GameWebSocketHandler:
    """Handles WebSocket messages for game sessions."""

    def __init__(self, connection_manager: ConnectionManager):
        self.manager = connection_manager
        self.game_manager = connection_manager.game_manager

    async def handle_message(self, websocket: WebSocket, game_id: str, data: dict):
        """
        Handle an incoming WebSocket message.

        Args:
            websocket: The WebSocket connection
            game_id: Game ID
            data: Parsed JSON message
        """
        msg_type = data.get("type")

        session = self.game_manager.get_session(game_id)
        if not session:
            await self.send_error(websocket, "Game not found")
            return

        handlers = {
            "get_state": self.handle_get_state,
            "get_moves": self.handle_get_moves,
            "move": self.handle_move,
            "ai_move": self.handle_ai_move,
            "start_ai_game": self.handle_start_ai_game,
            "pause": self.handle_pause,
            "resume": self.handle_resume,
            "step": self.handle_step,
            "ping": self.handle_ping,
        }

        handler = handlers.get(msg_type)
        if handler:
            await handler(websocket, session, data)
        else:
            await self.send_error(websocket, f"Unknown message type: {msg_type}")

    async def handle_get_state(self, websocket: WebSocket, session: GameSession, data: dict):
        """Send current game state."""
        state = self.game_manager.get_game_state(session)
        await self.manager.send_personal(websocket, {
            "type": "state",
            "state": state
        })

    async def handle_get_moves(self, websocket: WebSocket, session: GameSession, data: dict):
        """Send legal moves for current player."""
        moves = self.game_manager.get_legal_moves(session)
        await self.manager.send_personal(websocket, {
            "type": "legal_moves",
            "moves": moves
        })

    async def handle_move(self, websocket: WebSocket, session: GameSession, data: dict):
        """Handle a player making a move."""
        from_pos = tuple(data.get("from_pos", []))
        to_pos = tuple(data.get("to_pos", []))
        card = data.get("card", "")

        if len(from_pos) != 2 or len(to_pos) != 2:
            await self.send_error(websocket, "Invalid position format")
            return

        result = self.game_manager.make_move(session, from_pos, to_pos, card)

        if not result["success"]:
            await self.send_error(websocket, result.get("error", "Move failed"))
            return

        # Broadcast move to all observers
        await self.manager.broadcast(session.game_id, {
            "type": "move_made",
            "player": session.game.get_current_player(),
            "from_pos": list(from_pos),
            "to_pos": list(to_pos),
            "card": card,
            "capture": result.get("capture")
        })

        # Broadcast updated state
        await self.manager.broadcast(session.game_id, {
            "type": "state",
            "state": result["state"]
        })

        # Check for game over
        if result.get("game_over"):
            await self.manager.broadcast(session.game_id, {
                "type": "game_over",
                "winner": result.get("winner"),
                "reason": result.get("win_reason")
            })

    async def handle_ai_move(self, websocket: WebSocket, session: GameSession, data: dict):
        """Request AI to make a move."""
        if session.is_human_turn():
            await self.send_error(websocket, "It's the human player's turn")
            return

        result = await self.game_manager.make_ai_move(session)

        if not result:
            await self.send_error(websocket, "AI could not make a move")
            return

        if not result["success"]:
            await self.send_error(websocket, result.get("error", "AI move failed"))
            return

        # Get the last move from history
        if session.game.move_history:
            player, from_pos, to_pos, card = session.game.move_history[-1]
            await self.manager.broadcast(session.game_id, {
                "type": "move_made",
                "player": player,
                "from_pos": list(from_pos),
                "to_pos": list(to_pos),
                "card": card,
                "capture": result.get("capture")
            })

        # Broadcast updated state
        await self.manager.broadcast(session.game_id, {
            "type": "state",
            "state": result["state"]
        })

        if result.get("game_over"):
            await self.manager.broadcast(session.game_id, {
                "type": "game_over",
                "winner": result.get("winner"),
                "reason": result.get("win_reason")
            })

    async def handle_start_ai_game(self, websocket: WebSocket, session: GameSession, data: dict):
        """Start an AI vs AI game."""
        if session.blue_agent_type == "human" or session.red_agent_type == "human":
            await self.send_error(websocket, "Cannot auto-play with human players")
            return

        speed = data.get("speed", 1.0)
        session.state = SessionState.PLAYING

        async def on_move(result):
            if session.game.move_history:
                player, from_pos, to_pos, card = session.game.move_history[-1]
                await self.manager.broadcast(session.game_id, {
                    "type": "move_made",
                    "player": player,
                    "from_pos": list(from_pos),
                    "to_pos": list(to_pos),
                    "card": card,
                    "capture": result.get("capture")
                })

            await self.manager.broadcast(session.game_id, {
                "type": "state",
                "state": result["state"]
            })

            if result.get("game_over"):
                await self.manager.broadcast(session.game_id, {
                    "type": "game_over",
                    "winner": result.get("winner"),
                    "reason": result.get("win_reason")
                })

        # Run AI game in background
        session.ai_task = asyncio.create_task(
            self.game_manager.run_ai_game(session, speed, on_move)
        )

        await self.manager.send_personal(websocket, {
            "type": "ai_game_started",
            "speed": speed
        })

    async def handle_pause(self, websocket: WebSocket, session: GameSession, data: dict):
        """Pause an AI game."""
        self.game_manager.pause_game(session)
        await self.manager.broadcast(session.game_id, {
            "type": "game_paused"
        })

    async def handle_resume(self, websocket: WebSocket, session: GameSession, data: dict):
        """Resume a paused AI game."""
        self.game_manager.resume_game(session)
        await self.manager.broadcast(session.game_id, {
            "type": "game_resumed"
        })

    async def handle_step(self, websocket: WebSocket, session: GameSession, data: dict):
        """Step one move in a paused game."""
        if session.state != SessionState.PAUSED:
            await self.send_error(websocket, "Game must be paused to step")
            return

        result = await self.game_manager.step_game(session)

        if not result:
            await self.send_error(websocket, "Could not step game")
            return

        if session.game.move_history:
            player, from_pos, to_pos, card = session.game.move_history[-1]
            await self.manager.broadcast(session.game_id, {
                "type": "move_made",
                "player": player,
                "from_pos": list(from_pos),
                "to_pos": list(to_pos),
                "card": card,
                "capture": result.get("capture")
            })

        await self.manager.broadcast(session.game_id, {
            "type": "state",
            "state": result["state"]
        })

        if result.get("game_over"):
            await self.manager.broadcast(session.game_id, {
                "type": "game_over",
                "winner": result.get("winner"),
                "reason": result.get("win_reason")
            })

    async def handle_ping(self, websocket: WebSocket, session: GameSession, data: dict):
        """Respond to ping."""
        await self.manager.send_personal(websocket, {"type": "pong"})

    async def send_error(self, websocket: WebSocket, message: str):
        """Send an error message."""
        await self.manager.send_personal(websocket, {
            "type": "error",
            "message": message
        })
