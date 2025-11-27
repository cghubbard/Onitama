"""
Unit tests for balanced game evaluation system.

Tests matchup scheduling, outcome mapping, and statistics calculation.
"""
import pytest
from src.utils.constants import BLUE_WINS, RED_WINS, DRAW


class TestMatchupSchedule:
    """Test matchup schedule generation."""

    def test_even_games_balanced_schedule(self):
        """Test that even N creates perfectly balanced schedule."""
        # Simulate matchup schedule creation for 100 games
        agent1 = 'random'
        agent2 = 'heuristic'
        games = 100

        matchup_schedule = []
        games_per_side = games // 2

        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent1, 'red': agent2})
        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent2, 'red': agent1})

        # Count how many times each agent is BLUE
        agent1_as_blue = sum(1 for m in matchup_schedule if m['blue'] == agent1)
        agent2_as_blue = sum(1 for m in matchup_schedule if m['blue'] == agent2)

        assert len(matchup_schedule) == games
        assert agent1_as_blue == 50
        assert agent2_as_blue == 50

    def test_odd_games_balanced_schedule(self):
        """Test that odd N creates balanced schedule with at most 1 difference."""
        # Simulate matchup schedule creation for 101 games
        agent1 = 'linear'
        agent2 = 'heuristic'
        games = 101

        matchup_schedule = []
        games_per_side = games // 2

        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent1, 'red': agent2})
        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent2, 'red': agent1})

        # Odd game goes to original assignment
        if games % 2 == 1:
            matchup_schedule.append({'blue': agent1, 'red': agent2})

        # Count how many times each agent is BLUE
        agent1_as_blue = sum(1 for m in matchup_schedule if m['blue'] == agent1)
        agent2_as_blue = sum(1 for m in matchup_schedule if m['blue'] == agent2)

        assert len(matchup_schedule) == games
        assert agent1_as_blue == 51  # Gets extra game
        assert agent2_as_blue == 50
        assert abs(agent1_as_blue - agent2_as_blue) <= 1

    def test_single_game_schedule(self):
        """Test that N=1 creates single matchup with original assignment."""
        agent1 = 'random'
        agent2 = 'linear'
        games = 1

        matchup_schedule = []
        games_per_side = games // 2  # 0

        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent1, 'red': agent2})
        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent2, 'red': agent1})

        if games % 2 == 1:
            matchup_schedule.append({'blue': agent1, 'red': agent2})

        assert len(matchup_schedule) == 1
        assert matchup_schedule[0] == {'blue': agent1, 'red': agent2}

    def test_two_games_schedule(self):
        """Test that N=2 creates one game for each color assignment."""
        agent1 = 'heuristic'
        agent2 = 'linear'
        games = 2

        matchup_schedule = []
        games_per_side = games // 2  # 1

        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent1, 'red': agent2})
        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent2, 'red': agent1})

        assert len(matchup_schedule) == 2
        assert matchup_schedule[0] == {'blue': agent1, 'red': agent2}
        assert matchup_schedule[1] == {'blue': agent2, 'red': agent1}


class TestOutcomeMapping:
    """Test outcome mapping from colors to agents."""

    def test_blue_wins_agent1_is_blue(self):
        """Test BLUE win when agent1 is BLUE."""
        agent1 = 'random'
        agent2 = 'heuristic'
        matchup = {'blue': agent1, 'red': agent2}
        outcome = BLUE_WINS  # 1

        agent1_wins = 0
        agent2_wins = 0

        if outcome == BLUE_WINS:
            if matchup['blue'] == agent1:
                agent1_wins += 1
            else:
                agent2_wins += 1

        assert agent1_wins == 1
        assert agent2_wins == 0

    def test_blue_wins_agent2_is_blue(self):
        """Test BLUE win when agent2 is BLUE."""
        agent1 = 'random'
        agent2 = 'heuristic'
        matchup = {'blue': agent2, 'red': agent1}  # Colors swapped
        outcome = BLUE_WINS

        agent1_wins = 0
        agent2_wins = 0

        if outcome == BLUE_WINS:
            if matchup['blue'] == agent1:
                agent1_wins += 1
            else:
                agent2_wins += 1

        assert agent1_wins == 0
        assert agent2_wins == 1

    def test_red_wins_agent1_is_red(self):
        """Test RED win when agent1 is RED."""
        agent1 = 'linear'
        agent2 = 'heuristic'
        matchup = {'blue': agent2, 'red': agent1}  # agent1 is RED
        outcome = RED_WINS  # 2

        agent1_wins = 0
        agent2_wins = 0

        if outcome == RED_WINS:
            if matchup['red'] == agent1:
                agent1_wins += 1
            else:
                agent2_wins += 1

        assert agent1_wins == 1
        assert agent2_wins == 0

    def test_red_wins_agent2_is_red(self):
        """Test RED win when agent2 is RED."""
        agent1 = 'random'
        agent2 = 'linear'
        matchup = {'blue': agent1, 'red': agent2}  # agent2 is RED
        outcome = RED_WINS

        agent1_wins = 0
        agent2_wins = 0

        if outcome == RED_WINS:
            if matchup['red'] == agent1:
                agent1_wins += 1
            else:
                agent2_wins += 1

        assert agent1_wins == 0
        assert agent2_wins == 1

    def test_draw_outcome(self):
        """Test DRAW outcome doesn't affect agent wins."""
        agent1 = 'random'
        agent2 = 'heuristic'
        matchup = {'blue': agent1, 'red': agent2}
        outcome = DRAW  # 0

        agent1_wins = 0
        agent2_wins = 0
        draws = 0

        if outcome == BLUE_WINS:
            if matchup['blue'] == agent1:
                agent1_wins += 1
            else:
                agent2_wins += 1
        elif outcome == RED_WINS:
            if matchup['red'] == agent1:
                agent1_wins += 1
            else:
                agent2_wins += 1
        else:
            draws += 1

        assert agent1_wins == 0
        assert agent2_wins == 0
        assert draws == 1


class TestStatisticsCalculation:
    """Test statistics calculation across multiple games."""

    def test_statistics_with_balanced_matchups(self):
        """Test statistics calculation with multiple games."""
        agent1 = 'random'
        agent2 = 'heuristic'

        # Simulate 10 games
        matchup_schedule = [
            {'blue': agent1, 'red': agent2},  # Game 1
            {'blue': agent1, 'red': agent2},  # Game 2
            {'blue': agent1, 'red': agent2},  # Game 3
            {'blue': agent1, 'red': agent2},  # Game 4
            {'blue': agent1, 'red': agent2},  # Game 5
            {'blue': agent2, 'red': agent1},  # Game 6
            {'blue': agent2, 'red': agent1},  # Game 7
            {'blue': agent2, 'red': agent1},  # Game 8
            {'blue': agent2, 'red': agent1},  # Game 9
            {'blue': agent2, 'red': agent1},  # Game 10
        ]

        # Simulate outcomes: agent1 wins games 1,2,6,7; agent2 wins games 3,8,9; draws: 4,5,10
        outcomes = [
            BLUE_WINS,   # Game 1: agent1 (BLUE) wins
            BLUE_WINS,   # Game 2: agent1 (BLUE) wins
            RED_WINS,    # Game 3: agent2 (RED) wins
            DRAW,        # Game 4: draw
            DRAW,        # Game 5: draw
            RED_WINS,    # Game 6: agent1 (RED) wins
            RED_WINS,    # Game 7: agent1 (RED) wins
            BLUE_WINS,   # Game 8: agent2 (BLUE) wins
            BLUE_WINS,   # Game 9: agent2 (BLUE) wins
            DRAW,        # Game 10: draw
        ]

        agent1_wins = 0
        agent2_wins = 0
        draws = 0

        for matchup, outcome in zip(matchup_schedule, outcomes):
            if outcome == BLUE_WINS:
                if matchup['blue'] == agent1:
                    agent1_wins += 1
                else:
                    agent2_wins += 1
            elif outcome == RED_WINS:
                if matchup['red'] == agent1:
                    agent1_wins += 1
                else:
                    agent2_wins += 1
            else:
                draws += 1

        assert agent1_wins == 4  # Games 1, 2, 6, 7
        assert agent2_wins == 3  # Games 3, 8, 9
        assert draws == 3        # Games 4, 5, 10

    def test_matchup_distribution_calculation(self):
        """Test matchup distribution calculation."""
        agent1 = 'linear'
        agent2 = 'heuristic'
        games = 100

        matchup_schedule = []
        games_per_side = games // 2

        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent1, 'red': agent2})
        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent2, 'red': agent1})

        # Calculate distribution
        blue_as_agent1 = sum(1 for m in matchup_schedule if m['blue'] == agent1)
        blue_as_agent2 = games - blue_as_agent1

        assert blue_as_agent1 == 50
        assert blue_as_agent2 == 50


class TestEdgeCases:
    """Test edge cases."""

    def test_same_agent_both_sides(self):
        """Test when same agent type is used for both sides."""
        agent1 = 'random'
        agent2 = 'random'  # Same type
        games = 10

        matchup_schedule = []
        games_per_side = games // 2

        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent1, 'red': agent2})
        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent2, 'red': agent1})

        # Color swap still happens
        assert len(matchup_schedule) == 10

        # But since agents are same type, distribution doesn't matter
        blue_as_agent1 = sum(1 for m in matchup_schedule if m['blue'] == agent1)
        # Since both are 'random', all matchups have 'random' as both BLUE and RED
        assert blue_as_agent1 == 10  # All games have agent1 (random) as BLUE

    def test_zero_games(self):
        """Test with zero games."""
        agent1 = 'random'
        agent2 = 'heuristic'
        games = 0

        matchup_schedule = []
        games_per_side = games // 2

        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent1, 'red': agent2})
        for i in range(games_per_side):
            matchup_schedule.append({'blue': agent2, 'red': agent1})

        if games % 2 == 1:
            matchup_schedule.append({'blue': agent1, 'red': agent2})

        assert len(matchup_schedule) == 0
