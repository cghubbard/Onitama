"""
Pydantic models for the evaluation sandbox API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class FeatureBreakdown(BaseModel):
    """Single feature's contribution to the evaluation."""
    feature_name: str
    feature_value: float
    weight: float
    contribution: float  # weight * value


class MoveEvaluation(BaseModel):
    """Evaluation result for a single legal move."""
    from_pos: List[int]
    to_pos: List[int]
    card: str
    total_score: float
    features: List[FeatureBreakdown]
    is_winning_move: bool = False
    is_capture: bool = False


class PositionEvaluationRequest(BaseModel):
    """Request to evaluate a position."""
    game_id: str
    move_number: int
    weights: Optional[List[float]] = Field(
        default=None,
        description="Custom weights for the 11 features. Uses defaults if not provided."
    )


class PositionEvaluationResponse(BaseModel):
    """Full position evaluation response."""
    game_id: str
    move_number: int
    current_player: int
    current_position_score: float
    current_position_features: List[FeatureBreakdown]
    moves: List[MoveEvaluation]
    weights_used: List[float]
    feature_names: List[str]


class EvaluationModel(BaseModel):
    """Metadata about an evaluation model."""
    id: str
    name: str
    description: str
    feature_names: List[str]
    default_weights: List[float]


class ModelsResponse(BaseModel):
    """List of available evaluation models."""
    models: List[EvaluationModel]
