"""
Data structures for game trajectories.

These structures capture all information needed for:
- Game replay
- RL training (states, actions, legal moves)
- Analysis and debugging
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


@dataclass
class StateSnapshot:
    """
    Complete game state at a single point in time.

    Captures everything needed to reconstruct the game position
    and understand what decisions were available.
    """
    # Board state: "x,y" -> [player_id, piece_type]
    board: Dict[str, List[int]]

    # Current player (0=BLUE, 1=RED)
    current_player: int

    # Cards held by each player
    blue_cards: List[str]
    red_cards: List[str]
    neutral_card: str

    # Game progress
    move_number: int
    outcome: int  # 0=ONGOING, 1=BLUE_WINS, 2=RED_WINS, 3=DRAW

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Transition:
    """
    Single state transition in a game.

    Captures the state before an action, the action taken,
    and all legal moves that were available. This is the
    fundamental unit for RL training data.
    """
    # Move number (0-indexed)
    move_number: int

    # State before the action
    state: StateSnapshot

    # All legal moves available at this state
    legal_moves: List[Dict[str, Any]]

    # Action that was taken: {"from": [x,y], "to": [x,y], "card": "name"}
    # None for terminal states (no action possible from absorbing state)
    action: Optional[Dict[str, Any]] = None

    # Capture info (None if no capture, or [player, piece_type] of captured piece)
    capture: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "move_number": self.move_number,
            "state": self.state.to_dict(),
            "legal_moves": self.legal_moves,
            "action": self.action,
            "capture": self.capture
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transition':
        """Create from dictionary."""
        return cls(
            move_number=data["move_number"],
            state=StateSnapshot.from_dict(data["state"]),
            legal_moves=data["legal_moves"],
            action=data["action"],
            capture=data.get("capture")
        )


@dataclass
class GameConfig:
    """Configuration for a game session."""
    cards_used: List[str]
    blue_agent: str
    red_agent: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameConfig':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GameOutcome:
    """Final outcome of a game."""
    winner: Optional[int]  # 0=BLUE, 1=RED, None=DRAW
    reason: str  # "master_captured", "shrine_reached", "draw", "max_moves"
    total_moves: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameOutcome':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GameTrajectory:
    """
    Complete record of a single game.

    Contains all information needed to:
    - Replay the game move by move
    - Train RL agents (state-action sequences)
    - Analyze game patterns
    """
    # Identification
    game_id: str
    timestamp: str
    version: str = "1.0"

    # Configuration
    config: GameConfig = None

    # Complete trajectory of transitions
    transitions: List[Transition] = field(default_factory=list)

    # Final outcome (None until game ends)
    outcome: Optional[GameOutcome] = None

    def __post_init__(self):
        """Generate game_id and timestamp if not provided."""
        if not self.game_id:
            self.game_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def add_transition(self, transition: Transition):
        """Add a transition to the trajectory."""
        self.transitions.append(transition)

    def set_outcome(self, winner: Optional[int], reason: str):
        """Set the final game outcome."""
        self.outcome = GameOutcome(
            winner=winner,
            reason=reason,
            total_moves=len(self.transitions)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "game_id": self.game_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "config": self.config.to_dict() if self.config else None,
            "transitions": [t.to_dict() for t in self.transitions],
            "outcome": self.outcome.to_dict() if self.outcome else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameTrajectory':
        """Create from dictionary."""
        trajectory = cls(
            game_id=data["game_id"],
            timestamp=data["timestamp"],
            version=data.get("version", "1.0")
        )

        if data.get("config"):
            trajectory.config = GameConfig.from_dict(data["config"])

        trajectory.transitions = [
            Transition.from_dict(t) for t in data.get("transitions", [])
        ]

        if data.get("outcome"):
            trajectory.outcome = GameOutcome.from_dict(data["outcome"])

        return trajectory

    def get_state_at_move(self, move_number: int) -> Optional[StateSnapshot]:
        """Get the state at a specific move number."""
        if 0 <= move_number < len(self.transitions):
            return self.transitions[move_number].state
        return None

    def get_final_state(self) -> Optional[StateSnapshot]:
        """Get the final state of the game."""
        if self.transitions:
            return self.transitions[-1].state
        return None
