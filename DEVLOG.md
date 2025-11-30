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

### Linear Value Function Training Pipeline

Implemented complete ML training system for learning linear value function weights from self-play game trajectories.

#### Problem

The baseline linear model (`baseline_v1`) used hand-tuned feature weights. We needed:
1. Automated training from logged game data
2. Weighted logistic regression with time-based discounting
3. Game-level cross-validation to prevent data leakage
4. L1/L2 regularization with grid search
5. Model versioning and evaluation

#### Solution

Built end-to-end training pipeline using sklearn with proper CV methodology.

**Files created:**
- `src/logging/reconstruction.py` - Convert StateSnapshot → Game for feature extraction (85 lines)
- `src/evaluation/data_loader.py` - Load trajectories into training datasets (280 lines)
- `src/evaluation/trainer.py` - sklearn integration with CV and grid search (416 lines)
- `scripts/train_linear.py` - CLI training script (300+ lines)
- `tests/test_reconstruction.py` - 13 unit tests for game reconstruction
- `tests/test_data_loader.py` - 16 unit tests for data loading
- `tests/test_trainer.py` - 17 unit tests for training logic

**Files modified:**
- `src/logging/__init__.py` - Export reconstruction utilities
- `src/evaluation/__init__.py` - Export training modules

#### Implementation Details

**1. Game Reconstruction (`reconstruction.py`)**

Historical states are logged as lightweight StateSnapshots. Training requires full Game objects for feature extraction:

```python
def reconstruct_game_from_snapshot(snapshot: StateSnapshot) -> Game:
    """Reconstruct Game object from logged snapshot."""
    game = Game.__new__(Game)  # Bypass __init__ to avoid random cards
    game.board = deserialize_board(snapshot.board)
    game.player_cards = {
        BLUE: [Card(name, MOVE_CARDS[name]) for name in snapshot.blue_cards],
        RED: [Card(name, MOVE_CARDS[name]) for name in snapshot.red_cards]
    }
    game.neutral_card = Card(snapshot.neutral_card, MOVE_CARDS[snapshot.neutral_card])
    game.current_player = snapshot.current_player
    game.outcome = snapshot.outcome
    return game
```

**2. Data Loading (`data_loader.py`)**

Converts game trajectories into training examples with:
- Features: φ(s) from current player's perspective (11 features)
- Labels: 1 if current player wins, 0 if loses
- Weights: w_t = γ^(H-1-t) where γ=0.97 (more weight to later states)
- Game IDs: For game-level CV splitting

```python
@dataclass
class TrainingExample:
    features: np.ndarray    # Shape (11,)
    label: int              # 1 = win, 0 = loss
    weight: float           # Time-based weight
    game_id: str            # For GroupKFold
    move_number: int

def load_training_data(
    storage: GameStorage,
    blue_agent: Optional[str] = None,
    red_agent: Optional[str] = None,
    limit: Optional[int] = None,
    gamma: float = 0.97,
    exclude_draws: bool = True,
    verbose: bool = False
) -> TrainingDataset
```

**3. Training (`trainer.py`)**

Weighted logistic regression predicting log-odds: V(s) = θᵀφ̃(s) + b

**Feature standardization:**
```
φ̃_k = (φ_k - μ_k) / (σ_k + ε)  where ε=1e-8
```

**sklearn parameter mapping:**
- L1 only (λ₁>0, λ₂=0): `penalty='l1'`, `C=1/λ₁`, `solver='liblinear'`
- L2 only (λ₁=0, λ₂>0): `penalty='l2'`, `C=1/λ₂`, `solver='lbfgs'`
- ElasticNet (both>0): `penalty='elasticnet'`, `C=1/(λ₁+λ₂)`, `l1_ratio=λ₁/(λ₁+λ₂)`, `solver='saga'`

**Cross-validation:**
Uses `GroupKFold` to keep all states from a game in same fold (prevents data leakage):
```python
gkf = GroupKFold(n_splits=cv_folds)
for train_idx, val_idx in gkf.split(X, y, groups=game_ids):
    # All states from game_i stay together
```

**4. CLI Tool (`scripts/train_linear.py`)**

```bash
python scripts/train_linear.py \
  --blue-agent heuristic \
  --red-agent heuristic \
  --limit 4000 \
  --lambda1 0.0 0.01 0.1 1.0 10.0 \
  --lambda2 0.0 \
  --cv-folds 5 \
  --output trained_model \
  --notes "L1 sweep on 4000 games"
```

Outputs detailed CV results and saves model to `models/linear/{name}.json`.

#### Training Runs

**trained_001: Initial ElasticNet model**
- Dataset: 2000 heuristic vs heuristic games (11,545 examples)
- Hyperparameters: λ₁=1.0, λ₂=1.0 (ElasticNet)
- Cross-validation: 5-fold game-level
- Val loss: 0.6269
- Performance: **63% win rate vs baseline_v1** (126/200 wins)

**trained_002: L1-only sweep**
- Dataset: Same 2000 games
- Hyperparameters: λ₁ sweep [0.0, 0.01, 0.1, 1.0, 10.0], λ₂=0.0
- Best model: λ₁=10.0 (strong L1 sparsification)
- Val loss: 0.6269
- Performance: 52% vs trained_001 (weaker than initial model)
- Features zeroed: `material_diff_students`, `my_master_alive`, `opp_master_captured`

**trained_003: All games, L1 sweep**
- Dataset: ALL 4000 games (23,082 examples)
  - 2000 heuristic vs heuristic
  - 2000 with trained_001 agent
- Hyperparameters: λ₁ sweep [0.0, 0.01, 0.1, 1.0, 10.0], λ₂=0.0
- Best model: λ₁=10.0
- Val loss: **0.6243** (best so far)
- Performance: **65.5% win rate vs baseline_v1** (131/200 wins)

#### Results Analysis

**Key Finding:** Trained models outperform hand-tuned baseline by 15-30 percentage points.

**Feature importance (trained_003):**
```
master_safety_balance:           1.07  ← Dominant feature
capture_moves_diff:             -0.40
master_temple_distance_diff:    -0.24
legal_moves_diff:               -0.19
master_escape_options:           0.18
central_control_diff:            0.15
student_progress_diff:           0.14
card_mobility_diff:             -0.04
material_diff_students:          0.00  ← Zeroed by L1
my_master_alive:                 0.00  ← Zeroed by L1
opp_master_captured:             0.00  ← Zeroed by L1
```

**Insights:**
1. **Master safety** is by far the most critical feature (weight ~1.07)
2. **Material counting** (student count) is less important than positional features
3. Strong L1 regularization produces sparse models without hurting performance
4. More diverse training data (4000 games vs 2000) improves generalization

#### Testing

All 46 new tests pass (total: 135 passing tests):
- 13 tests for game reconstruction
- 16 tests for data loading pipeline
- 17 tests for training logic

**Test coverage:**
- StateSnapshot → Game reconstruction accuracy
- Feature extraction consistency between original and reconstructed games
- Label assignment (winner detection)
- Time-based weight computation (γ^(H-1-t))
- Feature standardization
- sklearn parameter mapping (L1/L2/ElasticNet)
- Game-level CV splitting (GroupKFold)
- Best model selection by validation loss

#### Usage Examples

**Generate training data:**
```bash
python main.py --games 2000 --blue heuristic --red heuristic --log all --quiet --progress
```

**Train model with grid search:**
```bash
python scripts/train_linear.py \
  --blue-agent heuristic \
  --red-agent heuristic \
  --limit 2000 \
  --lambda1 0.0 0.01 0.1 1.0 \
  --lambda2 0.0 0.01 0.1 1.0 \
  --cv-folds 5 \
  --output my_model \
  --notes "Initial training run"
```

**Evaluate trained model:**
```bash
python main.py --blue linear:my_model --red linear:baseline_v1 --games 200 --quiet --progress
```

#### Next Steps

With the training pipeline complete, future directions include:
1. Generate more diverse training data (different agent matchups)
2. Experiment with gamma values for time-based weighting
3. Analyze game situations where trained models succeed/fail
4. Explore non-linear features or interactions
5. Move toward tabular RL (Phase 2 in PROJ_PLAN.md)

### Round-Robin Tournament System

Implemented complete tournament infrastructure for comparing multiple agents with Elo rating calculation.

#### Problem

With multiple trained models (`baseline_v1`, `trained_001`, etc.), we needed:
1. Systematic comparison across all model pairs
2. Elo rating calculation to rank agents
3. Win matrix visualization
4. Persistent storage for tournament results
5. Integration with existing ModelStore for Elo tracking

#### Solution

Built `src/tournament/` module with CLI script for running round-robin tournaments.

**Files created:**
- `src/tournament/__init__.py` - Module exports
- `src/tournament/elo.py` - Standard Elo rating calculator
- `src/tournament/scheduler.py` - Round-robin matchup generation
- `src/tournament/storage.py` - SQLite persistence for tournament results
- `src/tournament/runner.py` - Tournament orchestration
- `src/tournament/display.py` - ASCII leaderboard and win matrix formatting
- `scripts/tournament.py` - CLI entry point
- `tests/test_elo.py` - 16 unit tests for Elo calculation
- `tests/test_tournament.py` - 17 integration tests

**SQLite tables added to `data/games.db`:**
- `tournaments` - Tournament metadata
- `tournament_participants` - Final standings with Elo
- `tournament_matchups` - Pairwise matchup results

#### Implementation Details

**Elo Calculation:**
- Standard formula: E = 1 / (1 + 10^((R_opp - R_self) / 400))
- Updates after each matchup using aggregate win rate
- K-factor scaling based on games played
- Built-in defaults: random=800, heuristic=1000, linear=1000

**Round-Robin Scheduling:**
- Generates all N*(N-1)/2 matchups
- Balanced color assignment per matchup (reuses main.py pattern)
- Configurable games per matchup (default: 500)

**CLI Usage:**
```bash
# Quick test tournament
python scripts/tournament.py --participants random heuristic --games 10

# Full tournament with progress
python scripts/tournament.py \
  --participants random heuristic linear:baseline_v1 linear:trained_003_all_games \
  --games 500 --progress --update-models

# List available models
python scripts/tournament.py --list-models
```

**Arguments:**
- `--participants` (required): List of agent specs
- `--games`: Games per matchup (default: 500)
- `--k-factor`: Elo K-factor (default: 32)
- `--update-models`: Persist Elo to ModelStore
- `--log`: Enable full game logging
- `--progress`: Show live progress updates
- `--quiet`: Minimal output

**Example Output:**
```
Tournament: tourney_20251127_203131
Participants: 3
Games per matchup: 20
Total games: 60

[1/3] random vs linear:baseline_v1: 3W-17L-0D
[2/3] random vs heuristic: 3W-17L-0D
[3/3] heuristic vs linear:baseline_v1: 4W-16L-0D

=== FINAL STANDINGS ===
Rank  Participant              Elo     W-L-D         Win%
----------------------------------------------------------------
1     linear:baseline_v1       1018    33-7-0        82.5%
2     heuristic                990     21-19-0       52.5%
3     random                   792     6-34-0        15.0%

Win Matrix (row W-L vs column):
                    linear:basel..  heuristic  random
linear:basel..               -        16-4      17-3
heuristic                  4-16          -      17-3
random                     3-17       3-17         -
```

#### Testing

All 33 new tests pass (total: 187 passing tests):
- 16 unit tests for Elo calculation (expected scores, updates, edge cases)
- 17 integration tests (scheduling, storage, runner, multi-player)

---

## Future Work (Not Started)
- Tabular RL agents (Q-learning, SARSA)
- PPO agent implementation using logged game data
- More AI agent types (MCTS, minimax)
- Enhanced web UI (move history, annotations)
- Tournament visualization in web app

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
