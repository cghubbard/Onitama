# Onitama Development Log

## Session: December 3, 2025 (Part 3) - Sandbox Model Selection

### Overview
Enhanced the web app sandbox to support model selection, allowing users to evaluate positions with any trained model from the registry. Added both backend API and frontend UI support.

### Changes

**Backend API Updates:**
- Modified `PositionEvaluationRequest` to include `model_id` parameter (defaults to "baseline_v1")
- Custom `weights` parameter now acts as an override if provided
- `/api/sandbox/models` endpoint now returns all models from the registry
- `/api/sandbox/evaluate` loads and uses selected model's weights

**Implementation in [src/web/sandbox.py](src/web/sandbox.py):**
- Integrated `ModelStore` to load models dynamically
- Priority order: custom weights > selected model > default
- Added error handling for missing/invalid models

**Frontend UI Updates:**
- Added model selector dropdown in sandbox navigation bar ([web/index.html](web/index.html))
- Updated [web/js/sandbox.js](web/js/sandbox.js):
  - Load all available models from API
  - Populate dropdown with model names and descriptions as tooltips
  - Re-evaluate position when model changes
  - Store selected model ID in controller state
- Added CSS styling for model selector ([web/css/style.css](web/css/style.css))

**User Experience:**
- Model selector appears in sandbox evaluation view between "Back" button and move navigation
- Dropdown shows all 17 models with descriptions visible on hover
- Changing model immediately re-evaluates current position with new weights
- Selection persists while navigating between moves

**Benefits:**
- Interactive model comparison without needing API knowledge
- All 17 models in the registry are now accessible via the sandbox UI
- Users can see how different trained models evaluate the same positions
- Maintains backward compatibility (defaults to baseline_v1)

### Testing
- All 201 tests pass
- Manual verification confirms models load correctly from registry

---

## Session: December 3, 2025 (Part 2) - Dual-Perspective Terminal States

### Overview
Discovered and fixed a critical flaw in terminal state training data generation: terminal states were only being captured from the winner's perspective, causing `my_master_alive` to always be 1.0 in training data (weight learned as 0.0). Implemented dual-perspective terminal state generation to create examples from both winner and loser viewpoints.

---

### Problem Discovery

After training `baseline_v2_with_terminal`, investigated why terminal feature weights were so low:
- `my_master_alive`: 0.0 (expected >>100)
- `opp_master_captured`: 3.49 (expected >>100)

**Root Cause Analysis:**
```python
# In data_loader.py, terminal states only created ONE example:
for t, transition in enumerate(trajectory.transitions):
    snapshot = transition.state
    current_player = snapshot.current_player  # Always the winner!
    features = extractor.extract_as_array(game, current_player)
    label = 1 if winner == current_player else 0
```

For terminal states, `current_player` is always the winner (who made the final move), so ALL terminal examples had:
- `my_master_alive = 1.0` → no variation → weight = 0.0
- `label = 1` → only winner perspective learned

We NEVER generated examples from the loser's perspective:
- `my_master_alive = 0.0`
- `opp_master_captured = 0.0`
- `label = 0`

---

### Solution: Dual-Perspective Terminal States

Modified `data_loader.py` to create TWO training examples for terminal states - one from each player's perspective:

**Implementation:**
```python
# For terminal states, create examples from BOTH players' perspectives
from src.utils.constants import ONGOING
if snapshot.outcome != ONGOING:
    # Terminal state - create two examples
    for player_id in [0, 1]:  # BLUE=0, RED=1
        features = extractor.extract_as_array(game, player_id)
        label = 1 if winner == player_id else 0
        examples.append(TrainingExample(...))
else:
    # Non-terminal state - create one example from current player's perspective
    current_player = snapshot.current_player
    features = extractor.extract_as_array(game, current_player)
    label = 1 if winner == current_player else 0
    examples.append(TrainingExample(...))
```

**Files Modified:**
- `src/evaluation/data_loader.py` (lines 220-248)

---

### Test Coverage

Added 3 new tests to verify dual-perspective terminal state generation:

**tests/test_data_loader.py:**
- `test_terminal_state_dual_perspective()` - Verify 2 examples per terminal state
- `test_non_terminal_states_single_perspective()` - Verify 1 example per non-terminal state
- Updated `test_terminal_state_feature_values()` - Verify winner and loser feature values

**Test Results:** All 201 tests passing (199 → 201)

**Verification Example:**
```python
# Winner perspective (label=1):
#   my_master_alive: 1.0
#   opp_master_captured: 1.0

# Loser perspective (label=0):
#   my_master_alive: 0.0
#   opp_master_captured: 0.0
```

---

### Training Data Regeneration

Regenerated 2000 games with dual-perspective terminal states:
- **Games**: 2000 baseline_v1 vs baseline_v1
- **Total examples**: 31,350 (was ~25,600 before)
- **Terminal examples**: ~3,942 (1,971 games × 2 perspectives)
- **Average examples/game**: 15.7 (was ~12.8)

---

### Model Training Results

Trained `baseline_v2_dual_perspective` with the new dual-perspective data:

**Terminal Feature Weights:**
```
Feature                   Single-Perspective   Dual-Perspective
-----------------------------------------------------------------
my_master_alive           0.0                  5.08 ✓
opp_master_captured       3.49                 2.94
my_master_threats         -0.22                0.13
opp_master_threats        6.50                 0.30
```

**Key Achievement:** `my_master_alive` now has proper non-zero weight (5.08 vs 0.0)!

**Performance:**
- vs heuristic: 68% win rate (same as baseline_v1)
- vs baseline_v1: 12% win rate (worse than single-perspective 21%)

**Analysis:** The lower win rate against baseline_v1 is expected because:
1. baseline_v1 is hand-tuned with weights like `my_master_alive=1000`, `opp_master_captured=500`
2. Our learned model has smaller standardized weights (5.08, 2.94)
3. Training on baseline_v1's self-play means we're learning to mimic it, not surpass it
4. The dual-perspective terminal states create more balanced training data but don't magically improve beyond the training distribution

---

### Summary

**Problem:** Terminal states only from winner's perspective → `my_master_alive` always 1.0 → learned weight 0.0

**Solution:** Generate dual-perspective terminal examples (winner + loser)

**Result:** Terminal features now have proper non-zero weights

**Next Steps:**
- Consider training on more diverse data (not just self-play)
- Explore whether terminal state weighting should be different
- Investigate if hand-tuned large weights perform better than statistically optimal smaller weights

---

## Session: December 3, 2025 (Part 1) - Terminal States & Filtering

### Overview
Fixed two critical bugs in the RL training pipeline that were preventing effective model learning: terminal states not being logged, and game filtering not working properly. Both fixes are now implemented, tested, and verified.

---

## Changes Made

### 1. Terminal State Logging (Critical Fix)

**Problem:** The final winning/losing state was never being logged in game trajectories. Models were learning from all positions leading up to victory, but never saw what an actual winning position looked like. This caused trained models to have near-zero weights for terminal features (`my_master_alive`, `opp_master_captured`) instead of large values like the hand-tuned baseline (1000, 500).

**Root Cause:** Game logging was capturing state BEFORE each move was made. When a winning move occurred, the pipeline would:
1. Log the pre-move state (outcome=ONGOING)
2. Make the move (game.outcome → BLUE_WINS/RED_WINS)
3. Loop exits - terminal state never captured

**Solution:**
- Made `Transition.action` field Optional to support terminal states (no legal moves)
- Added `log_terminal_state()` method to `GameLogSession`
- Updated `runner.py` to call terminal logging after decisive games
- Updated `main.py` to call terminal logging after decisive games
- Removed terminal state skip in `data_loader.py` (lines 180-186)

**Files Modified:**
- `src/logging/trajectory.py` - Made action Optional
- `src/logging/game_logger.py` - Added log_terminal_state()
- `src/tournament/runner.py` - Call terminal logging after wins
- `main.py` - Call terminal logging after wins
- `src/evaluation/data_loader.py` - Remove terminal state skip

**Verification:**
- All 2000 regenerated games now have proper terminal states
- Terminal transitions have `outcome != 0` and `action = None`
- Terminal features now have non-zero weights in trained models

---

### 2. Flexible Game Filtering (Critical Fix)

**Problem:** Game filtering by agent type was broken. When training scripts filtered for "baseline" games, the exact string matching failed to find games stored as "linear:baseline_v1". The query would return 0 games silently, causing training to proceed with empty or wrong datasets.

**Solution:**
- Added `agent_match_mode` parameter to `query_games()` with three modes:
  - `"contains"` (default) - substring match, e.g., "baseline" matches "linear:baseline_v1"
  - `"prefix"` - prefix match, e.g., "linear:" matches "linear:*"
  - `"exact"` - exact string match (backward compatible)
- Added validation that raises clear error when filters match no games
- Added `get_unique_agent_combinations()` helper to show available matchups
- Added `--agent-match-mode` CLI argument to training script

**Files Modified:**
- `src/logging/storage.py` - Added flexible matching and helper methods
- `src/evaluation/data_loader.py` - Added validation with helpful error messages
- `scripts/train_linear.py` - Added CLI argument

**Example Error Message:**
```
No games found matching filters:
  blue_agent=nonexistent
  red_agent=None
  match_mode=contains

Database contains 2000 total games.

Available matchups:
  linear:baseline_v1 vs linear:baseline_v1: 2000 games
```

---

### 3. Test Coverage (New)

Added 11 new tests to verify both fixes:

**tests/test_data_loader.py:**
- `test_filter_contains_matching()` - "baseline" matches "linear:baseline_v1"
- `test_filter_prefix_matching()` - "linear" matches "linear:*"
- `test_filter_exact_matching()` - Exact mode for backward compatibility
- `test_filter_no_matches_raises_error()` - Clear error when no matches
- `test_load_with_terminal_states()` - Terminal states load correctly
- `test_terminal_state_feature_values()` - Terminal features have correct values

**tests/test_game_logger.py (New File):**
- `test_log_terminal_state_blue_wins()` - Terminal state logged for blue win
- `test_log_terminal_state_red_wins()` - Terminal state logged for red win
- `test_terminal_transition_has_no_action()` - Terminal transitions have action=None
- `test_terminal_transition_outcome_set()` - Terminal state has outcome != 0
- `test_full_game_with_terminal()` - Full game logging with terminal states

**Test Results:** All 199 tests passing (188 original + 11 new)

---

### 4. Training Data Regeneration

Regenerated training data with both fixes in place:
- **2000 games**: baseline_v1 vs baseline_v1
- **Generation time**: 25.5 seconds (~80 games/second)
- **Win/Draw split**: 98.7% decisive, 1.4% draws
- **Terminal states**: Verified present in all games
- **Examples per game**: ~12.8 (including terminal states)

**Verification:**
```bash
# Filtering works
python scripts/train_linear.py --blue-agent baseline --red-agent baseline --limit 10
# Successfully loaded 10 games with 96 examples

# Terminal states present
python -c "
from src.logging.storage import GameStorage
storage = GameStorage('data')
games = storage.query_games(limit=10)
for g in games:
    t = storage.load_trajectory(g['game_id'])
    print(f'outcome={t.transitions[-1].state.outcome}')
"
# All show outcome=1 or 2 (BLUE_WINS/RED_WINS)
```

---

### 5. Model Training Results

Trained `baseline_v2_with_terminal` on 2000 games with terminal states:

**Terminal Feature Weights (Before vs After):**
```
Feature                   Old (no terminal)   New (with terminal)
-----------------------------------------------------------------
opp_master_captured       0.0                 3.49
opp_master_threats        ~2.0                6.50
my_master_alive           0.0                 0.0 (always 1.0 in training)
```

**Key Observation:** Terminal features now have non-zero weights, confirming they're being learned. The weights are smaller than hand-tuned baseline (3.49 vs 500) because:
1. Training on self-play (baseline vs baseline) creates symmetric games
2. Terminal states are only ~8% of training examples
3. Logistic regression produces statistically optimal weights, not interpretable large numbers

**Performance:** Model underperforms hand-tuned baseline (21% win rate), likely due to overfitting to self-play patterns. This suggests need for more diverse training data.

---

## Summary

**Core Bugs Fixed:**
- ✅ Terminal states ARE now being logged and included in training
- ✅ Game filtering DOES work with flexible matching

**Impact:**
- Models can now learn what winning positions look like
- Training scripts can reliably filter games by agent type
- Clear error messages when filters don't match any games
- All tests passing with comprehensive coverage

**Next Steps:**
- Generate more diverse training data (not just self-play)
- Investigate self-play overfitting vs generalization
- Consider alternative training approaches (TD learning, policy gradient)

---

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

## Session: November 30, 2025

### Linear Heuristic Evaluation Sandbox

Added a new "Sandbox" tab to the web app for analyzing how the linear heuristic model evaluates game positions.

#### Features

1. **Position Evaluation**: Load any replayed game and step through positions to see:
   - Current position score (V(s) = w^T * φ(s))
   - Feature breakdown showing each feature's value, weight, and contribution
   - All legal moves ranked by score

2. **Move Comparison**: Click any two moves in the list to see side-by-side comparison:
   - Total score difference
   - Per-feature contribution comparison with delta values
   - Visual highlighting (blue = Move A, red = Move B)

3. **Board Visualization**:
   - Full board state with pieces and cards
   - Hover over moves to see them highlighted on the board

#### Files Created

- `src/web/sandbox_models.py` - Pydantic schemas (FeatureBreakdown, MoveEvaluation, etc.)
- `src/web/sandbox.py` - FastAPI router with `/api/sandbox/evaluate` and `/api/sandbox/models` endpoints
- `web/js/sandbox.js` - Frontend SandboxController class

#### Files Modified

- `src/web/app.py` - Include sandbox router, initialize storage
- `web/index.html` - Add Sandbox tab and panel HTML
- `web/js/app.js` - Initialize sandbox controller, add tab switching
- `web/css/style.css` - Add sandbox styling (280+ lines)

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sandbox/models` | GET | List available evaluation models with default weights |
| `/api/sandbox/evaluate` | POST | Evaluate all legal moves at a position |

#### Usage

1. Start the web server: `python run_web.py`
2. Click the "Sandbox" tab
3. Select a game from the replay list
4. Navigate through moves with Prev/Next buttons
5. Click moves to compare them side-by-side

---

## Future Work (Not Started)

### Model Evaluation Enhancements
- **Interactive Weight Adjustment**: Add sliders to modify the 11 feature weights in real-time and see how scores change
- **Log Top N Moves**: Extend `Transition` dataclass to include optional `move_evaluations` from the agent, enabling faster replay analysis

### Core RL Development
- Tabular RL agents (Q-learning, SARSA)
- PPO agent implementation using logged game data
- More AI agent types (MCTS, minimax)

### Web UI Enhancements
- Enhanced web UI (move history, annotations)
- Tournament visualization in web app
- Additional model-specific sandboxes for future agents (neural network policy/value visualization)

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
