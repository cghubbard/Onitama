"""
Sandbox API for evaluating positions with different models.

Provides endpoints for:
- Listing available evaluation models
- Evaluating all legal moves at a specific game position
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from src.web.sandbox_models import (
    PositionEvaluationRequest, PositionEvaluationResponse,
    MoveEvaluation, FeatureBreakdown, ModelsResponse, EvaluationModel
)
from src.logging.storage import GameStorage
from src.logging.reconstruction import reconstruct_game_from_snapshot
from src.logging.trajectory import GameTrajectory
from src.evaluation.features import FeatureExtractor
from src.evaluation.weights import FEATURE_NAMES
from src.evaluation.model_store import ModelStore
from src.utils.constants import BLUE_WINS, RED_WINS, BLUE, RED, ONGOING

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])

# Global storage instance (set during startup)
_storage: Optional[GameStorage] = None


def set_storage(storage: GameStorage):
    """Set the storage instance for the sandbox router."""
    global _storage
    _storage = storage


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available evaluation models from the model registry."""
    store = ModelStore()
    registry_entries = store.list_models()

    models = []
    for entry in registry_entries:
        try:
            model = store.load(entry.name)
            models.append(EvaluationModel(
                id=model.name,
                name=model.name,
                description=entry.notes or f"Model trained on {model.training.num_games if model.training else 0} games",
                feature_names=FEATURE_NAMES,
                default_weights=model.get_weight_vector()
            ))
        except Exception as e:
            # Skip models that fail to load
            print(f"Warning: Could not load model {entry.name}: {e}")
            continue

    return ModelsResponse(models=models)


@router.post("/evaluate", response_model=PositionEvaluationResponse)
async def evaluate_position(request: PositionEvaluationRequest):
    """
    Evaluate all legal moves at a specific game position.

    Reconstructs the game state from a saved trajectory and computes
    the linear heuristic evaluation for each legal move.
    """
    if _storage is None:
        raise HTTPException(status_code=500, detail="Storage not initialized")

    # Load trajectory
    trajectory = _storage.load_trajectory(request.game_id)
    if not trajectory:
        raise HTTPException(status_code=404, detail="Game not found")

    # Handle both dict and GameTrajectory object
    if isinstance(trajectory, dict):
        trajectory = GameTrajectory.from_dict(trajectory)

    if request.move_number < 0 or request.move_number >= len(trajectory.transitions):
        raise HTTPException(status_code=400, detail="Invalid move number")

    # Get the transition at this move
    transition = trajectory.transitions[request.move_number]

    # Reconstruct game state
    game = reconstruct_game_from_snapshot(transition.state)

    # Determine weights to use: custom weights > model weights > defaults
    if request.weights:
        weights = request.weights
        if len(weights) != 16:
            raise HTTPException(status_code=400, detail="Weights must have exactly 16 values")
    else:
        # Load model weights
        store = ModelStore()
        model_id = request.model_id or "baseline_v1"
        try:
            model = store.load(model_id)
            weights = model.get_weight_vector()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    extractor = FeatureExtractor()
    current_player = transition.state.current_player

    # Evaluate current position
    current_features = extractor.extract_as_array(game, current_player)
    current_score = sum(w * f for w, f in zip(weights, current_features))

    current_breakdown = [
        FeatureBreakdown(
            feature_name=name,
            feature_value=float(current_features[i]),
            weight=float(weights[i]),
            contribution=float(weights[i] * current_features[i])
        )
        for i, name in enumerate(FEATURE_NAMES)
    ]

    # Get legal moves and board state
    legal_moves = game.get_legal_moves(current_player)
    board = game.get_board_state()

    # Evaluate each legal move
    move_evaluations = []
    for from_pos, to_pos, card_name in legal_moves:
        # Simulate the move
        simulated = game.copy()
        simulated.make_move(from_pos, to_pos, card_name)

        # Check for winning move
        outcome = simulated.get_outcome()
        is_winning = False
        if outcome != ONGOING:
            if current_player == BLUE and outcome == BLUE_WINS:
                is_winning = True
            elif current_player == RED and outcome == RED_WINS:
                is_winning = True

        # Check if capture
        is_capture = to_pos in board and board[to_pos][0] != current_player

        # Extract features after move
        if is_winning:
            total_score = 999999.0  # Large sentinel value for winning moves (JSON-compliant)
            features_after = [0.0] * 16
        else:
            features_after = extractor.extract_as_array(simulated, current_player)
            total_score = sum(w * f for w, f in zip(weights, features_after))

        feature_breakdown = [
            FeatureBreakdown(
                feature_name=name,
                feature_value=float(features_after[i]),
                weight=float(weights[i]),
                contribution=float(weights[i] * features_after[i])
            )
            for i, name in enumerate(FEATURE_NAMES)
        ]

        move_evaluations.append(MoveEvaluation(
            from_pos=list(from_pos),
            to_pos=list(to_pos),
            card=card_name,
            total_score=total_score,
            features=feature_breakdown,
            is_winning_move=is_winning,
            is_capture=is_capture
        ))

    # Sort moves by score (best first)
    move_evaluations.sort(key=lambda m: m.total_score, reverse=True)

    return PositionEvaluationResponse(
        game_id=request.game_id,
        move_number=request.move_number,
        current_player=current_player,
        current_position_score=current_score,
        current_position_features=current_breakdown,
        moves=move_evaluations,
        weights_used=weights,
        feature_names=FEATURE_NAMES
    )
