# Onitama Development Log

## Session: November 2025

### Overview
Picked up the project after a long absence. Built a web interface for playing/observing games and implemented RL-ready game logging.

---

## Changes Made

### 1. Web Interface (New)
Built a complete web UI for playing Onitama against AI agents.

**Files created:**
- `src/web/app.py` - FastAPI application with REST and WebSocket endpoints
- `src/web/models.py` - Pydantic request/response models
- `src/web/game_manager.py` - Game session management
- `src/web/websocket.py` - Real-time game communication
- `run_web.py` - Web server entry point
- `web/index.html` - Main HTML page
- `web/js/app.js` - Frontend application logic
- `web/js/game.js` - Game state management (GameState, GameController)
- `web/js/board.js` - Board and card rendering (BoardRenderer, CardRenderer)
- `web/js/websocket.js` - WebSocket client
- `web/css/style.css` - Styling

**Features:**
- Human vs AI gameplay
- AI vs AI observation with pause/resume/step controls
- Game replay browser with move-by-move navigation
- Real-time updates via WebSocket

**To run:**
```bash
python run_web.py --port 8765
# Then open http://127.0.0.1:8765
```

---

### 2. Game Logging System (New)
Implemented RL-ready logging that captures raw game data for future training.

**Files created:**
- `src/logging/trajectory.py` - Data structures (StateSnapshot, Transition, GameTrajectory)
- `src/logging/storage.py` - JSONL + SQLite storage
- `src/logging/game_logger.py` - Logging interface with configurable modes
- `src/game/serialization.py` - Game state serialization utilities

**Design philosophy:** Log raw data (states, actions, legal moves, captures). RL training code computes rewards later - keeps logging simple and flexible.

**Storage:**
- `data/games/YYYY-MM-DD.jsonl` - Full game trajectories
- `data/games.db` - SQLite index for fast querying

**Usage:**
```bash
# Log all games
python main.py --games 100 --log all --quiet

# Sample 20% of games
python main.py --games 1000 --log sample --sample-rate 0.2 --quiet

# No logging
python main.py --games 10 --log none
```

---

### 3. Card Movement Fixes (Bug Fix)
Fixed incorrect card movement definitions that had forward/backward swapped.

**File modified:** `src/utils/constants.py`

**Root cause:** Card movements were defined with forward/backward inverted. For example, Dragon's wide diagonal moves were going backward instead of forward.

**Cards fixed:** Dragon, Elephant, Goose, Rooster, Mantis, Crane, and others audited against official Onitama card patterns.

**How movements work:**
- Movements are defined from RED's perspective (sitting at y=4, facing y=0)
- Negative dy = forward (toward opponent)
- Positive dy = backward (toward own side)
- BLUE gets movements negated by `Card.get_movements(player)`

---

### 4. Bug Fixes in Web UI

**AI not responding after human move:**
- Fixed in `web/js/game.js` - Added automatic AI move request in state handler

**Replay info persisting when returning to Play tab:**
- Fixed in `web/js/app.js` - Clear replay controls and info in `showSetup()`

---

### 5. Cleanup
- Removed `main_patched.py` (was identical duplicate of `main.py`)
- `main_debug.py` kept for reference (uses signal-based timeout approach)

---

## Architecture Notes

### Coordinate System
- Board is 5x5, coordinates are (x, y) where x=column, y=row
- BLUE starts at y=0 (top), RED starts at y=4 (bottom)
- BLUE_SHRINE = (2, 0), RED_SHRINE = (2, 4)

### Card Movement Transformation
```
constants.py    →    Card class    →    Game logic
(RED perspective)    (negates for BLUE)   (applies to board)
```

### Game State Flow (Web)
```
Frontend ←WebSocket→ GameWebSocketHandler → GameManager → Game
                           ↓
                     GameLogger → Storage (JSONL + SQLite)
```

---

## Dependencies Added
```
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
websockets>=11.0
pydantic>=2.0
```

---

## Session: November 26, 2025

### Linear Heuristic Agent & Feature Extraction

Implemented a linear evaluation function V(s) = w^T * φ(s) with 11 hand-crafted features for board position evaluation.

**Files created:**
- `src/evaluation/__init__.py` - Module exports
- `src/evaluation/features.py` - FeatureExtractor class with 11 features
- `src/evaluation/weights.py` - Default weight vector and feature names
- `src/agents/linear_heuristic_agent.py` - New agent using 1-ply lookahead
- `tests/test_features.py` - 28 unit tests for feature extraction
- `tests/test_web.py` - 48 unit tests for web app components

**Features implemented:**
1. `material_diff_students` - Student count difference
2. `my_master_alive` - Binary flag for own master survival
3. `opp_master_captured` - Binary flag for capturing opponent master
4. `master_safety_balance` - Threats to masters comparison
5. `legal_moves_diff` - Mobility difference
6. `capture_moves_diff` - Available capture moves difference
7. `master_temple_distance_diff` - Progress toward enemy shrine
8. `student_progress_diff` - Average student advancement
9. `central_control_diff` - Pieces in central 3x3 area
10. `card_mobility_diff` - Total moves from held cards
11. `master_escape_options` - Safe squares for own master

**Files modified:**
- `src/game/game.py` - Added `copy()` method for move simulation
- `src/agents/__init__.py` - Export LinearHeuristicAgent
- `main.py` - Added 'linear' agent type to CLI

**Usage:**
```bash
# Run tournament between agents
python main.py --games 100 --blue heuristic --red linear --quiet --progress

# With logging for replay
python main.py --games 100 --blue linear --red heuristic --log all --quiet
```

### Replay Move Highlighting

Added visual highlighting in replay mode to show piece movement.

**Files modified:**
- `web/js/board.js` - Added `highlightReplayMove(fromPos, toPos)` method
- `web/js/app.js` - Call highlighting after rendering replay state; fixed duplicate event listeners bug
- `web/css/style.css` - Added `.replay-from` (blue) and `.replay-to` (green) CSS classes
- `web/index.html` - Moved replay slider to correct location in DOM

**Bug fixes:**
- Fixed replay controls skipping moves due to duplicate event listeners
- Fixed slider not visible during replay (was in hidden panel)

---

## Session: November 27, 2025

### Balanced Game Evaluation System

Implemented balanced evaluation for agent matchups to eliminate first-player advantage bias in tournament results.

#### Problem
When running multiple games for agent comparison:
- BLUE always went first (hardcoded first-player advantage)
- Same card set could be reused across all N games
- Statistics tracked by color (BLUE wins vs RED wins), not by agent
- This created systematic bias making fair agent comparison impossible

#### Solution
Modified `main.py` to swap agent-to-color assignment across games:
- For N games: first N/2 have agent1 as BLUE, next N/2 have agent2 as BLUE
- Each game uses fresh random cards (unless `--cards` specified)
- Statistics now track agent wins instead of color wins
- Maintains full backward compatibility

**Files modified:**
- `main.py` - Matchup scheduling, outcome-to-agent mapping, agent-centric statistics

**Files created:**
- `tests/test_balanced_evaluation.py` - 13 unit tests for matchup logic

#### Implementation Details

**Matchup Schedule (lines 196-209):**
```python
# Create balanced matchup schedule
matchup_schedule = []
games_per_side = args.games // 2

for i in range(games_per_side):
    matchup_schedule.append({'blue': args.blue, 'red': args.red})
for i in range(games_per_side):
    matchup_schedule.append({'blue': args.red, 'red': args.blue})

# If odd number of games, give extra game to original color assignment
if args.games % 2 == 1:
    matchup_schedule.append({'blue': args.blue, 'red': args.red})
```

**Outcome Mapping (lines 251-263):**
Maps game outcomes to agent wins regardless of color assignment:
```python
if outcome == BLUE_WINS:
    if matchup['blue'] == args.blue:
        agent1_wins += 1
    else:
        agent2_wins += 1
elif outcome == RED_WINS:
    if matchup['red'] == args.blue:
        agent1_wins += 1
    else:
        agent2_wins += 1
```

**Statistics Display (lines 268-283):**
Shows agent wins and matchup distribution:
```
===== Results =====
Games played: 100
linear wins: 73 (73.0%)
heuristic wins: 27 (27.0%)
Draws: 0

Matchup distribution:
  linear as BLUE: 50/100
  heuristic as BLUE: 50/100
```

#### Usage

**Balanced evaluation (automatic):**
```bash
python main.py --blue linear --red heuristic --games 100 --quiet
# First 50 games: linear=BLUE, heuristic=RED
# Next 50 games: heuristic=BLUE, linear=RED
```

**With specific cards (still balanced):**
```bash
python main.py --blue random --red linear --games 50 --cards Tiger Dragon Frog Rabbit Crab --quiet
# All games use same cards, but colors alternate
```

**Single game (unchanged behavior):**
```bash
python main.py --blue heuristic --red linear
# Original behavior: args.blue always BLUE
```

#### Testing

Added comprehensive unit tests covering:
- Matchup schedule generation (even N, odd N, N=1, N=2)
- Outcome-to-agent mapping (all outcome types)
- Statistics calculation with multiple games
- Edge cases (same agent both sides, zero games)

All 89 tests pass (76 existing + 13 new).

#### Backward Compatibility

✅ Single game (N=1): Identical to previous behavior
✅ All CLI flags work unchanged
✅ Game class untouched (per CLAUDE.md constraints)
✅ Existing scripts/notebooks continue working

---

## Future Work (Not Started)
- PPO agent implementation using logged game data
- More AI agent types (MCTS, minimax)
- Enhanced web UI (move history, annotations)

---

## Quick Reference

**Start web server:**
```bash
python run_web.py --port 8765
```

**Run CLI games with logging:**
```bash
python main.py --games 100 --blue heuristic --red heuristic --log all --quiet
```

**Kill web server:**
```bash
pkill -f "python run_web.py"
```

**Key files to understand the codebase:**
- `src/game/game.py` - Core game logic
- `src/game/card.py` - Card movement handling
- `src/utils/constants.py` - All constants including card definitions
- `src/agents/heuristic_agent.py` - Current best AI agent
