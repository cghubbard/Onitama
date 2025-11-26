"""
Pydantic models for the Onitama web API.

Defines request/response schemas for REST endpoints and WebSocket messages.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import IntEnum


class Player(IntEnum):
    """Player identifiers."""
    BLUE = 0
    RED = 1


class PieceType(IntEnum):
    """Piece type identifiers."""
    PAWN = 0
    MASTER = 1


class GameOutcome(IntEnum):
    """Game outcome states."""
    ONGOING = 0
    BLUE_WINS = 1
    RED_WINS = 2
    DRAW = 3


class Position(BaseModel):
    """Board position."""
    x: int = Field(ge=0, le=4)
    y: int = Field(ge=0, le=4)


class Piece(BaseModel):
    """A piece on the board."""
    player: int
    piece_type: int
    position: Position


class CardInfo(BaseModel):
    """Card information with movement patterns."""
    name: str
    movements: List[List[int]]  # List of [dx, dy] movements


class GameConfig(BaseModel):
    """Configuration for creating a new game."""
    blue_agent: str = Field(default="human", description="Agent type: human, random, or heuristic")
    red_agent: str = Field(default="heuristic", description="Agent type: human, random, or heuristic")
    cards: Optional[List[str]] = Field(default=None, description="Optional list of 5 card names")


class MoveRequest(BaseModel):
    """Request to make a move."""
    from_pos: Position
    to_pos: Position
    card_name: str


class LegalMove(BaseModel):
    """A legal move."""
    from_pos: List[int]  # [x, y]
    to_pos: List[int]    # [x, y]
    card: str


class GameState(BaseModel):
    """Complete game state for API responses."""
    game_id: str
    board: Dict[str, List[int]]  # "x,y" -> [player, piece_type]
    current_player: int
    blue_cards: List[str]
    red_cards: List[str]
    neutral_card: str
    outcome: int
    move_count: int
    blue_agent: str
    red_agent: str
    last_move: Optional[Dict[str, Any]] = None


class GameSummary(BaseModel):
    """Summary of a game for listings."""
    game_id: str
    timestamp: str
    blue_agent: str
    red_agent: str
    winner: Optional[int]
    total_moves: int
    cards_used: List[str]


class GameListResponse(BaseModel):
    """Response for game list queries."""
    games: List[GameSummary]
    total: int
    limit: int
    offset: int


class CreateGameResponse(BaseModel):
    """Response after creating a game."""
    game_id: str
    state: GameState


class MoveResponse(BaseModel):
    """Response after making a move."""
    success: bool
    state: GameState
    game_over: bool
    winner: Optional[int] = None
    win_reason: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


# WebSocket message models

class WSMessage(BaseModel):
    """Base WebSocket message."""
    type: str


class WSJoinGame(WSMessage):
    """Client joining a game."""
    type: str = "join"
    game_id: str
    player: Optional[int] = None  # Which player to control (None = spectator)


class WSMakeMove(WSMessage):
    """Client making a move."""
    type: str = "move"
    from_pos: List[int]  # [x, y]
    to_pos: List[int]    # [x, y]
    card: str


class WSRequestAIMove(WSMessage):
    """Request AI to make a move."""
    type: str = "ai_move"


class WSStartAIGame(WSMessage):
    """Start an AI vs AI game."""
    type: str = "start_ai_game"
    speed: float = 1.0  # Seconds between moves


class WSPauseGame(WSMessage):
    """Pause an AI vs AI game."""
    type: str = "pause"


class WSResumeGame(WSMessage):
    """Resume an AI vs AI game."""
    type: str = "resume"


class WSStepGame(WSMessage):
    """Step one move in a paused AI game."""
    type: str = "step"


class WSGameState(WSMessage):
    """Server sending game state."""
    type: str = "state"
    state: GameState


class WSMoveMade(WSMessage):
    """Server notifying of move."""
    type: str = "move_made"
    player: int
    from_pos: List[int]
    to_pos: List[int]
    card: str
    capture: Optional[List[int]] = None  # [player, piece_type] if capture


class WSLegalMoves(WSMessage):
    """Server sending legal moves."""
    type: str = "legal_moves"
    moves: List[LegalMove]


class WSGameOver(WSMessage):
    """Server notifying game over."""
    type: str = "game_over"
    winner: Optional[int]
    reason: str


class WSError(WSMessage):
    """Server error message."""
    type: str = "error"
    message: str
