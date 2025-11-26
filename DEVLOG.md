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
