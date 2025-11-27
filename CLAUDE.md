# CLAUDE.md - Onitama RL Training Lab

## Project Overview

This is an implementation of the board game Onitama designed as a **reinforcement learning training lab**. The goal is to build progressively sophisticated AI agents, from simple heuristics through tabular RL to deep RL (PPO) and MCTS.

**What is Onitama?** A two-player strategy game on a 5x5 board. Each player controls a Master and 4 Students (pawns). Win by capturing the opponent's Master or moving your Master to their shrine (temple square).

## Development Priorities

1. **Primary: RL Agent Development** - Follow the phased roadmap in PROJ_PLAN.md
2. **Critical: Game Engine Correctness** - The foundation for all RL work; must be solid
3. **Low Priority: Web Interface** - Functional but not a focus area

## Quick Start

```bash
# Run tests (always do this before completing work)
pytest tests/

# Run balanced evaluation between agents (agents automatically alternate colors)
python main.py --blue linear --red heuristic --games 100 --quiet
# In 100 games: linear plays 50 as BLUE (first), 50 as RED
# Statistics show agent wins, not color wins

# Single game (original behavior, no color swapping)
python main.py --blue random --red heuristic

# Available agents: random, heuristic, linear

# Start web server (low priority, for observation/debugging)
python run_web.py --port 8000
```

## Project Structure

```
src/
├── game/           # Core game engine (be careful here)
│   ├── game.py     # Game state, rules, move validation
│   └── card.py     # Card representation
├── agents/         # AI agents (primary development area)
│   ├── agent.py    # Abstract base class
│   ├── random_agent.py
│   ├── heuristic_agent.py
│   ├── linear_heuristic_agent.py
│   └── ppo_agent.py  # Placeholder for future
├── evaluation/     # Feature extraction for RL
│   ├── features.py # 11-feature extractor
│   └── weights.py  # Weight vectors
├── logging/        # Game trajectory recording
└── web/            # FastAPI web interface (low priority)

tests/              # pytest test suites
data/               # Game logs (JSONL + SQLite)
```

## Key Files to Understand

- **PROJ_PLAN.md** - RL development roadmap (7 phases from heuristics to PPO)
- **DEVLOG.md** - Development history and session notes
- **src/agents/agent.py** - Agent interface (ABC with `select_move()`)
- **src/evaluation/features.py** - 11 heuristic features for position evaluation
- **src/utils/constants.py** - Board size, player IDs, card definitions

## Workflow Requirements

### Always Do:
- Run `pytest tests/` before considering any work complete
- Update DEVLOG.md with significant changes (new features, bug fixes, architectural changes)

### Off-Limits:
- **Card definitions in constants.py** - Never modify these
- **Game engine refactoring** - Avoid unless absolutely necessary. If you must touch `src/game/`, explain clearly why and get explicit approval

## Technical Context

### Coordinate System
- Board is 5x5, coordinates are `(x, y)` where x=column, y=row
- **BLUE player** starts at y=0 (bottom), shrine at (2, 4)
- **RED player** starts at y=4 (top), shrine at (2, 0)
- Card moves are defined from BLUE's perspective and transformed for RED

### Agent Interface
All agents extend the abstract `Agent` class:
```python
class Agent(ABC):
    def __init__(self, player_id: int): ...

    @abstractmethod
    def select_move(self, game: 'Game') -> Tuple[Tuple[int,int], Tuple[int,int], str]:
        """Returns (from_pos, to_pos, card_name)"""
```

### Feature System (11 features)
Used by LinearHeuristicAgent for evaluation V(s) = w^T * φ(s):
1. Material difference (student count)
2. My master alive
3. Opponent master captured
4. Master safety balance
5. Legal moves difference
6. Capture moves difference
7. Master-to-shrine distance
8. Student progress toward shrine
9. Central 3x3 control
10. Card mobility
11. Master escape options

### Game Logging
Games can be logged for RL training:
```bash
python main.py --games 100 --log all --quiet
# Saves to data/games/YYYY-MM-DD.jsonl
```

## Current State

See **PROJ_PLAN.md** for the full RL roadmap. Currently focusing on:
- Building out experimentation framework
- Strengthening evaluation methodology for comparing agents

## Known Gotchas

### Card Movement Transforms
Card moves are defined from BLUE's perspective. When applying to RED, the y-coordinates are negated. There was a historical bug where several cards had movements swapped - this is fixed but be aware of the coordinate transform pipeline.

### Circular Imports
Use `TYPE_CHECKING` blocks for type hints that would cause circular imports:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.game.game import Game
```

### Testing Edge Cases
When testing game states, you can construct custom board positions. See `tests/test_features.py` for examples of setting up specific scenarios.
