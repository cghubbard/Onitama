"""
Unit tests for the Elo rating calculator.
"""

import pytest
from src.tournament.elo import EloCalculator, EloUpdate


class TestEloCalculator:
    """Tests for EloCalculator class."""

    def test_initial_rating(self):
        """Test default initial rating."""
        calc = EloCalculator(initial_elo=1000)
        assert calc.get_rating("unknown_player") == 1000

    def test_builtin_defaults(self):
        """Test built-in agent default Elo values."""
        calc = EloCalculator()
        calc.set_initial_ratings(['random', 'heuristic', 'linear'])

        assert calc.get_rating('random') == 800
        assert calc.get_rating('heuristic') == 1000
        assert calc.get_rating('linear') == 1000

    def test_model_agent_defaults(self):
        """Test that model-based agents use linear default."""
        calc = EloCalculator(initial_elo=1000)
        calc.set_initial_ratings(['linear:baseline_v1', 'linear:trained_001'])

        # Should use linear default (1000)
        assert calc.get_rating('linear:baseline_v1') == 1000
        assert calc.get_rating('linear:trained_001') == 1000

    def test_existing_elos_override(self):
        """Test that existing Elo values are used when provided."""
        calc = EloCalculator()
        calc.set_initial_ratings(
            ['random', 'heuristic'],
            existing_elos={'random': 900, 'heuristic': 1100}
        )

        assert calc.get_rating('random') == 900
        assert calc.get_rating('heuristic') == 1100

    def test_expected_score_equal_ratings(self):
        """Test expected score when ratings are equal."""
        calc = EloCalculator()
        expected = calc.expected_score(1000, 1000)
        assert expected == pytest.approx(0.5, abs=0.001)

    def test_expected_score_higher_rating(self):
        """Test expected score when player A has higher rating."""
        calc = EloCalculator()
        expected = calc.expected_score(1200, 1000)
        # 200 point advantage -> ~76% expected
        assert expected > 0.7
        assert expected < 0.8

    def test_expected_score_lower_rating(self):
        """Test expected score when player A has lower rating."""
        calc = EloCalculator()
        expected = calc.expected_score(1000, 1200)
        # 200 point disadvantage -> ~24% expected
        assert expected > 0.2
        assert expected < 0.3

    def test_expected_score_symmetric(self):
        """Test that expected scores sum to 1."""
        calc = EloCalculator()
        expected_a = calc.expected_score(1100, 900)
        expected_b = calc.expected_score(900, 1100)
        assert expected_a + expected_b == pytest.approx(1.0, abs=0.001)

    def test_update_winner_gains_rating(self):
        """Test that winner gains rating points."""
        calc = EloCalculator(k_factor=32)
        calc.set_initial_ratings(['a', 'b'], {'a': 1000, 'b': 1000})

        result = calc.update_from_matchup('a', 'b', a_wins=10, b_wins=0, draws=0)

        assert result.new_elo_a > result.old_elo_a
        assert result.new_elo_b < result.old_elo_b
        assert result.delta_a > 0

    def test_update_loser_loses_rating(self):
        """Test that loser loses rating points."""
        calc = EloCalculator(k_factor=32)
        calc.set_initial_ratings(['a', 'b'], {'a': 1000, 'b': 1000})

        result = calc.update_from_matchup('a', 'b', a_wins=0, b_wins=10, draws=0)

        assert result.new_elo_a < result.old_elo_a
        assert result.new_elo_b > result.old_elo_b
        assert result.delta_a < 0

    def test_update_draws_minimal_change(self):
        """Test that draws between equal players cause minimal change."""
        calc = EloCalculator(k_factor=32)
        calc.set_initial_ratings(['a', 'b'], {'a': 1000, 'b': 1000})

        result = calc.update_from_matchup('a', 'b', a_wins=0, b_wins=0, draws=10)

        # With equal ratings and all draws, change should be minimal
        assert abs(result.delta_a) <= 1

    def test_update_zero_sum(self):
        """Test that Elo changes are zero-sum."""
        calc = EloCalculator(k_factor=32)
        calc.set_initial_ratings(['a', 'b'], {'a': 1000, 'b': 1000})

        result = calc.update_from_matchup('a', 'b', a_wins=7, b_wins=3, draws=0)

        delta_a = result.new_elo_a - result.old_elo_a
        delta_b = result.new_elo_b - result.old_elo_b
        assert delta_a + delta_b == 0

    def test_update_upset_larger_change(self):
        """Test that upsets cause larger rating changes."""
        calc = EloCalculator(k_factor=32)

        # Favorite wins (expected)
        calc.set_initial_ratings(['a', 'b'], {'a': 1200, 'b': 1000})
        expected_result = calc.update_from_matchup('a', 'b', a_wins=10, b_wins=0, draws=0)

        # Reset and try upset
        calc2 = EloCalculator(k_factor=32)
        calc2.set_initial_ratings(['a', 'b'], {'a': 1200, 'b': 1000})
        upset_result = calc2.update_from_matchup('a', 'b', a_wins=0, b_wins=10, draws=0)

        # Upset should cause larger magnitude change
        assert abs(upset_result.delta_a) > abs(expected_result.delta_a)

    def test_update_empty_matchup_raises(self):
        """Test that updating with zero games raises error."""
        calc = EloCalculator()
        calc.set_initial_ratings(['a', 'b'])

        with pytest.raises(ValueError):
            calc.update_from_matchup('a', 'b', a_wins=0, b_wins=0, draws=0)

    def test_get_all_ratings(self):
        """Test getting all ratings."""
        calc = EloCalculator()
        calc.set_initial_ratings(['a', 'b', 'c'], {'a': 1000, 'b': 1100, 'c': 900})

        ratings = calc.get_all_ratings()

        assert ratings == {'a': 1000, 'b': 1100, 'c': 900}
        # Verify it's a copy
        ratings['a'] = 9999
        assert calc.get_rating('a') == 1000


class TestEloUpdate:
    """Tests for EloUpdate dataclass."""

    def test_elo_update_creation(self):
        """Test EloUpdate dataclass creation."""
        update = EloUpdate(
            participant_a='player1',
            participant_b='player2',
            old_elo_a=1000,
            old_elo_b=1000,
            new_elo_a=1016,
            new_elo_b=984,
            expected_a=0.5,
            actual_score_a=0.7,
            delta_a=16
        )

        assert update.participant_a == 'player1'
        assert update.delta_a == 16
        assert update.new_elo_a - update.old_elo_a == 16
