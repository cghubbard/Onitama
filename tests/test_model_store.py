"""
Tests for model storage and CLI agent creation.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.evaluation.model_store import (
    ModelStore,
    LinearModel,
    TrainingInfo,
    NormalizationStats,
    create_baseline_model,
)
from src.evaluation.weights import FEATURE_NAMES, DEFAULT_WEIGHTS
from src.utils.constants import BLUE, RED
from main import parse_agent_spec, create_agent


class TestLinearModel:
    """Tests for LinearModel dataclass."""

    def test_create_model_with_weights(self):
        model = LinearModel(name="test", weights=DEFAULT_WEIGHTS.copy())
        assert model.name == "test"
        assert len(model.weights) == 16

    def test_get_weight_vector_order(self):
        """Weight vector should match FEATURE_NAMES order."""
        model = LinearModel(name="test", weights=DEFAULT_WEIGHTS.copy())
        vec = model.get_weight_vector()
        assert len(vec) == 16
        assert vec[0] == DEFAULT_WEIGHTS[FEATURE_NAMES[0]]

    def test_to_dict_roundtrip(self):
        """Model should survive serialization round-trip."""
        original = LinearModel(
            name="test_model",
            weights=DEFAULT_WEIGHTS.copy(),
            bias=0.5,
            training=TrainingInfo(num_games=100, lambda1=0.01),
        )
        data = original.to_dict()
        restored = LinearModel.from_dict(data)

        assert restored.name == original.name
        assert restored.bias == original.bias
        assert restored.weights == original.weights
        assert restored.training.num_games == 100
        assert restored.training.lambda1 == 0.01

    def test_baseline_model_factory(self):
        """create_baseline_model should return valid model."""
        model = create_baseline_model()
        assert model.name == "baseline_v1"
        assert model.elo == 1000
        assert len(model.get_weight_vector()) == 16


class TestModelStore:
    """Tests for ModelStore save/load/list operations."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary model store for testing."""
        temp_dir = tempfile.mkdtemp()
        store = ModelStore(models_dir=Path(temp_dir))
        yield store
        shutil.rmtree(temp_dir)

    def test_save_and_load(self, temp_store):
        """Should save and load model correctly."""
        model = LinearModel(name="test_save", weights=DEFAULT_WEIGHTS.copy())
        temp_store.save(model)

        loaded = temp_store.load("test_save")
        assert loaded.name == "test_save"
        assert loaded.weights == model.weights

    def test_list_models(self, temp_store):
        """Should list all saved models."""
        temp_store.save(LinearModel(name="model_a", weights=DEFAULT_WEIGHTS.copy()))
        temp_store.save(LinearModel(name="model_b", weights=DEFAULT_WEIGHTS.copy()))

        models = temp_store.list_models()
        names = [m.name for m in models]
        assert "model_a" in names
        assert "model_b" in names

    def test_exists(self, temp_store):
        """Should correctly report model existence."""
        assert not temp_store.exists("nonexistent")

        temp_store.save(LinearModel(name="exists", weights=DEFAULT_WEIGHTS.copy()))
        assert temp_store.exists("exists")

    def test_delete(self, temp_store):
        """Should delete model from store and registry."""
        temp_store.save(LinearModel(name="to_delete", weights=DEFAULT_WEIGHTS.copy()))
        assert temp_store.exists("to_delete")

        result = temp_store.delete("to_delete")
        assert result is True
        assert not temp_store.exists("to_delete")

    def test_load_nonexistent_raises(self, temp_store):
        """Loading nonexistent model should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            temp_store.load("does_not_exist")

    def test_update_elo(self, temp_store):
        """Should update Elo in both registry and model file."""
        temp_store.save(LinearModel(name="elo_test", weights=DEFAULT_WEIGHTS.copy(), elo=1000))

        temp_store.update_elo("elo_test", 1100)

        # Check registry
        models = temp_store.list_models()
        entry = next(m for m in models if m.name == "elo_test")
        assert entry.elo == 1100

        # Check model file
        loaded = temp_store.load("elo_test")
        assert loaded.elo == 1100


class TestParseAgentSpec:
    """Tests for parse_agent_spec function."""

    def test_simple_agent_types(self):
        assert parse_agent_spec("random") == ("random", None)
        assert parse_agent_spec("heuristic") == ("heuristic", None)
        assert parse_agent_spec("linear") == ("linear", None)

    def test_model_specification(self):
        assert parse_agent_spec("linear:baseline_v1") == ("linear", "baseline_v1")
        assert parse_agent_spec("linear:trained_001") == ("linear", "trained_001")

    def test_case_insensitive(self):
        assert parse_agent_spec("RANDOM") == ("random", None)
        assert parse_agent_spec("Linear:Model") == ("linear", "Model")


class TestCreateAgent:
    """Tests for create_agent function."""

    def test_create_random_agent(self):
        agent = create_agent("random", BLUE)
        assert agent.player_id == BLUE

    def test_create_heuristic_agent(self):
        agent = create_agent("heuristic", RED)
        assert agent.player_id == RED

    def test_create_linear_agent_default(self):
        agent = create_agent("linear", BLUE)
        assert agent.player_id == BLUE

    def test_create_linear_with_model(self):
        """Should load model weights when model name specified."""
        # This uses the real model store with baseline_v1
        agent = create_agent("linear:baseline_v1", BLUE)
        assert agent.player_id == BLUE
        # Verify it loaded the baseline weights
        assert agent.get_weights()[0] == DEFAULT_WEIGHTS[FEATURE_NAMES[0]]

    def test_unknown_agent_type_raises(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("unknown_type", BLUE)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not found"):
            create_agent("linear:nonexistent_model", BLUE)
