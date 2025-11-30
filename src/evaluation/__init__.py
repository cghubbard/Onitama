"""
Evaluation module for Onitama board positions.

Provides feature extraction and position evaluation for heuristic-based agents.
"""

from src.evaluation.features import FeatureExtractor, FeatureVector
from src.evaluation.weights import DEFAULT_WEIGHTS, DEFAULT_WEIGHT_VECTOR, FEATURE_NAMES
from src.evaluation.model_store import (
    ModelStore,
    LinearModel,
    TrainingInfo,
    NormalizationStats,
    create_baseline_model,
)
from src.evaluation.data_loader import (
    TrainingExample,
    TrainingDataset,
    load_training_data,
    split_by_games
)
from src.evaluation.trainer import (
    TrainingResult,
    GridSearchResults,
    train_linear_value_function,
    train_single_model
)

__all__ = [
    'FeatureExtractor',
    'FeatureVector',
    'DEFAULT_WEIGHTS',
    'DEFAULT_WEIGHT_VECTOR',
    'FEATURE_NAMES',
    'ModelStore',
    'LinearModel',
    'TrainingInfo',
    'NormalizationStats',
    'create_baseline_model',
    'TrainingExample',
    'TrainingDataset',
    'load_training_data',
    'split_by_games',
    'TrainingResult',
    'GridSearchResults',
    'train_linear_value_function',
    'train_single_model'
]
