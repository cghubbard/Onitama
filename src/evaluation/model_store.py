"""
Model storage and versioning for linear heuristic agents.

Provides functionality to save, load, and manage trained models with metadata.
Models are stored as JSON files with full reproducibility information.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field

from src.evaluation.weights import FEATURE_NAMES, DEFAULT_WEIGHTS


# Default paths
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"


@dataclass
class TrainingInfo:
    """Metadata about how a model was trained."""
    data_source: str = ""
    num_games: int = 0
    gamma: float = 0.97
    lambda1: float = 0.0
    lambda2: float = 0.0
    val_loss: Optional[float] = None
    train_loss: Optional[float] = None
    notes: str = ""


@dataclass
class NormalizationStats:
    """Feature normalization statistics."""
    means: List[float] = field(default_factory=lambda: [0.0] * 14)
    stds: List[float] = field(default_factory=lambda: [1.0] * 14)
    epsilon: float = 1e-8


@dataclass
class LinearModel:
    """
    A complete linear model with weights and metadata.

    Contains everything needed to use the model:
    - Weight vector and bias
    - Normalization statistics (for trained models)
    - Training metadata for reproducibility
    """
    name: str
    weights: Dict[str, float]
    bias: float = 0.0
    normalization: Optional[NormalizationStats] = None
    training: Optional[TrainingInfo] = None
    created: str = ""
    model_type: str = "linear"
    elo: Optional[int] = None

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        if self.normalization is None:
            self.normalization = NormalizationStats()
        if self.training is None:
            self.training = TrainingInfo()

    def get_weight_vector(self) -> List[float]:
        """Get weights as ordered list matching FEATURE_NAMES."""
        return [self.weights[name] for name in FEATURE_NAMES]

    def get_normalization_arrays(self) -> tuple:
        """Get normalization stats as (means, stds, epsilon)."""
        return (
            self.normalization.means,
            self.normalization.stds,
            self.normalization.epsilon
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "created": self.created,
            "elo": self.elo,
            "weights": self.weights,
            "bias": self.bias,
            "normalization": asdict(self.normalization) if self.normalization else None,
            "training": asdict(self.training) if self.training else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinearModel':
        """Create a LinearModel from a dictionary."""
        norm_data = data.get("normalization")
        train_data = data.get("training")

        return cls(
            name=data["name"],
            weights=data["weights"],
            bias=data.get("bias", 0.0),
            normalization=NormalizationStats(**norm_data) if norm_data else None,
            training=TrainingInfo(**train_data) if train_data else None,
            created=data.get("created", ""),
            model_type=data.get("model_type", "linear"),
            elo=data.get("elo"),
        )


@dataclass
class RegistryEntry:
    """Entry in the model registry."""
    name: str
    path: str
    elo: Optional[int] = None
    notes: str = ""


class ModelStore:
    """
    Manages storage and retrieval of linear models.

    Models are stored as JSON files in the models/ directory.
    A registry.json file tracks all available models.

    Usage:
        store = ModelStore()

        # Save a new model
        model = LinearModel(name="trained_001", weights={...})
        store.save(model)

        # Load a model by name
        model = store.load("trained_001")

        # List all available models
        models = store.list_models()
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the model store.

        Args:
            models_dir: Custom models directory (defaults to project models/)
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.registry_path = self.models_dir / "registry.json"
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "linear").mkdir(exist_ok=True)

        if not self.registry_path.exists():
            self._save_registry({"models": []})

    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"models": []}

    def _save_registry(self, registry: Dict[str, Any]):
        """Save the model registry."""
        with open(self.registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

    def _model_path(self, name: str) -> Path:
        """Get the file path for a model."""
        return self.models_dir / "linear" / f"{name}.json"

    def save(self, model: LinearModel, notes: str = "") -> Path:
        """
        Save a model to disk and register it.

        Args:
            model: The LinearModel to save
            notes: Optional notes for the registry entry

        Returns:
            Path to the saved model file
        """
        path = self._model_path(model.name)

        # Save the model file
        with open(path, 'w') as f:
            json.dump(model.to_dict(), f, indent=2)

        # Update registry
        registry = self._load_registry()

        # Remove existing entry if present
        registry["models"] = [
            m for m in registry["models"] if m["name"] != model.name
        ]

        # Add new entry
        relative_path = f"linear/{model.name}.json"
        registry["models"].append({
            "name": model.name,
            "path": relative_path,
            "elo": model.elo,
            "notes": notes or (model.training.notes if model.training else ""),
        })

        self._save_registry(registry)
        return path

    def load(self, name: str) -> LinearModel:
        """
        Load a model by name.

        Args:
            name: The model name

        Returns:
            The loaded LinearModel

        Raises:
            FileNotFoundError: If the model doesn't exist
        """
        path = self._model_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Model '{name}' not found at {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return LinearModel.from_dict(data)

    def exists(self, name: str) -> bool:
        """Check if a model exists."""
        return self._model_path(name).exists()

    def list_models(self) -> List[RegistryEntry]:
        """
        List all registered models.

        Returns:
            List of RegistryEntry objects
        """
        registry = self._load_registry()
        return [
            RegistryEntry(**entry) for entry in registry.get("models", [])
        ]

    def delete(self, name: str) -> bool:
        """
        Delete a model from disk and registry.

        Args:
            name: The model name to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._model_path(name)

        # Remove file if exists
        if path.exists():
            path.unlink()

        # Update registry
        registry = self._load_registry()
        original_count = len(registry["models"])
        registry["models"] = [
            m for m in registry["models"] if m["name"] != name
        ]

        if len(registry["models"]) < original_count:
            self._save_registry(registry)
            return True
        return False

    def update_elo(self, name: str, elo: int):
        """
        Update a model's Elo rating in the registry.

        Args:
            name: The model name
            elo: The new Elo rating
        """
        registry = self._load_registry()
        for entry in registry["models"]:
            if entry["name"] == name:
                entry["elo"] = elo
                break
        self._save_registry(registry)

        # Also update the model file
        if self.exists(name):
            model = self.load(name)
            model.elo = elo
            self.save(model)


def create_baseline_model() -> LinearModel:
    """
    Create the baseline model from hand-tuned weights.

    Returns:
        LinearModel with default hand-tuned weights
    """
    return LinearModel(
        name="baseline_v1",
        weights=DEFAULT_WEIGHTS.copy(),
        bias=0.0,
        training=TrainingInfo(
            notes="Hand-tuned baseline weights"
        ),
        elo=1000,  # Starting Elo
    )
