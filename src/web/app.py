"""
FastAPI application for Onitama web interface.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, List
import os

from src.web.models import (
    GameConfig, GameState, GameSummary, GameListResponse,
    CreateGameResponse, MoveRequest, MoveResponse, ErrorResponse,
    CardInfo
)
from src.web.game_manager import GameManager
from src.web.websocket import ConnectionManager, GameWebSocketHandler
from src.logging.storage import GameStorage
from src.game.serialization import serialize_all_cards

# Create FastAPI app
app = FastAPI(
    title="Onitama",
    description="Web interface for Onitama game with AI agents",
    version="1.0.0"
)

# Global instances (initialized in startup)
game_manager: Optional[GameManager] = None
storage: Optional[GameStorage] = None
connection_manager: Optional[ConnectionManager] = None
ws_handler: Optional[GameWebSocketHandler] = None


@app.on_event("startup")
async def startup():
    """Initialize global instances on startup."""
    global game_manager, storage, connection_manager, ws_handler

    storage = GameStorage(data_dir="data")
    game_manager = GameManager(data_dir="data")
    connection_manager = ConnectionManager(game_manager)
    ws_handler = GameWebSocketHandler(connection_manager)


# =============================================================================
# REST API Endpoints
# =============================================================================

@app.post("/api/games", response_model=CreateGameResponse)
async def create_game(config: GameConfig):
    """Create a new game session."""
    try:
        session = game_manager.create_game(
            blue_agent=config.blue_agent,
            red_agent=config.red_agent,
            cards=config.cards
        )
        state = game_manager.get_game_state(session)
        return CreateGameResponse(game_id=session.game_id, state=GameState(**state))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    """Get current game state."""
    session = game_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")

    state = game_manager.get_game_state(session)
    return {"state": state}


@app.get("/api/games/{game_id}/moves")
async def get_legal_moves(game_id: str):
    """Get legal moves for current player."""
    session = game_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")

    moves = game_manager.get_legal_moves(session)
    return {"moves": moves}


@app.post("/api/games/{game_id}/moves")
async def make_move(game_id: str, move: MoveRequest):
    """Make a move in a game."""
    session = game_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")

    from_pos = (move.from_pos.x, move.from_pos.y)
    to_pos = (move.to_pos.x, move.to_pos.y)

    result = game_manager.make_move(session, from_pos, to_pos, move.card_name)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Invalid move"))

    return result


@app.delete("/api/games/{game_id}")
async def end_game(game_id: str):
    """End a game session."""
    session = game_manager.get_session(game_id)
    if not session:
        raise HTTPException(status_code=404, detail="Game not found")

    game_manager.end_game(game_id, "user_ended")
    return {"status": "ended"}


@app.get("/api/games")
async def list_active_games():
    """List all active game sessions."""
    games = game_manager.list_active_games()
    return {"games": games}


# =============================================================================
# Replay API Endpoints
# =============================================================================

@app.get("/api/replays", response_model=GameListResponse)
async def list_replays(
    blue_agent: Optional[str] = None,
    red_agent: Optional[str] = None,
    winner: Optional[int] = None,
    limit: int = Query(default=50, le=100),
    offset: int = 0
):
    """List saved game replays with optional filters."""
    games = storage.query_games(
        blue_agent=blue_agent,
        red_agent=red_agent,
        winner=winner,
        limit=limit,
        offset=offset
    )

    total = storage.count_games(
        blue_agent=blue_agent,
        red_agent=red_agent,
        winner=winner
    )

    return GameListResponse(
        games=[GameSummary(**g) for g in games],
        total=total,
        limit=limit,
        offset=offset
    )


@app.get("/api/replays/{game_id}")
async def get_replay(game_id: str):
    """Get full game trajectory for replay."""
    trajectory = storage.load_trajectory(game_id)
    if not trajectory:
        raise HTTPException(status_code=404, detail="Replay not found")

    return trajectory.to_dict()


@app.get("/api/replays/{game_id}/state/{move_number}")
async def get_replay_state(game_id: str, move_number: int):
    """Get game state at a specific move."""
    trajectory = storage.load_trajectory(game_id)
    if not trajectory:
        raise HTTPException(status_code=404, detail="Replay not found")

    if move_number < 0 or move_number >= len(trajectory.transitions):
        raise HTTPException(status_code=400, detail="Invalid move number")

    transition = trajectory.transitions[move_number]
    return {
        "state": transition.state.to_dict(),
        "action": transition.action,
        "legal_moves": transition.legal_moves,
        "capture": transition.capture
    }


@app.get("/api/stats")
async def get_stats():
    """Get aggregate statistics about stored games."""
    return storage.get_statistics()


# =============================================================================
# Reference Data Endpoints
# =============================================================================

@app.get("/api/agents")
async def list_agents():
    """List available AI agents."""
    return {
        "agents": [
            {"id": "human", "name": "Human Player", "description": "Controlled by user"},
            {"id": "random", "name": "Random Agent", "description": "Makes random legal moves"},
            {"id": "heuristic", "name": "Heuristic Agent", "description": "Uses strategic heuristics"}
        ]
    }


@app.get("/api/cards")
async def list_cards():
    """List all available cards with their movements."""
    return {"cards": serialize_all_cards()}


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time game communication."""
    session = game_manager.get_session(game_id)
    if not session:
        await websocket.close(code=4004, reason="Game not found")
        return

    await connection_manager.connect(websocket, game_id)

    # Send initial state
    state = game_manager.get_game_state(session)
    await websocket.send_json({"type": "state", "state": state})

    try:
        while True:
            data = await websocket.receive_json()
            await ws_handler.handle_message(websocket, game_id, data)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        connection_manager.disconnect(websocket)


# =============================================================================
# Static Files (Frontend)
# =============================================================================

# Get the path to the web directory
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")


@app.get("/")
async def serve_index():
    """Serve the main page."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Onitama API. Frontend not found. Use /docs for API documentation."}


# Mount static files if the directory exists
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
