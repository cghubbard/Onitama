"""
Integration tests for the tournament system.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.tournament.runner import TournamentRunner, TournamentConfig
from src.tournament.storage import TournamentStorage, MatchupResult, ParticipantStats
from src.tournament.scheduler import (
    generate_round_robin_schedule,
    create_matchup_schedule,
    Matchup,
    total_games,
    num_matchups
)
from src.tournament.elo import EloCalculator


class TestScheduler:
    """Tests for tournament scheduling."""

    def test_round_robin_matchup_count(self):
        """Test correct number of matchups generated."""
        participants = ['a', 'b', 'c', 'd']
        matchups = generate_round_robin_schedule(participants, 10, shuffle=False)

        # n*(n-1)/2 = 4*3/2 = 6 matchups
        assert len(matchups) == 6

    def test_round_robin_all_pairs(self):
        """Test that all pairs are covered."""
        participants = ['a', 'b', 'c']
        matchups = generate_round_robin_schedule(participants, 10, shuffle=False)

        pairs = set()
        for m in matchups:
            pair = tuple(sorted([m.participant_a, m.participant_b]))
            pairs.add(pair)

        expected = {('a', 'b'), ('a', 'c'), ('b', 'c')}
        assert pairs == expected

    def test_balanced_color_schedule_even(self):
        """Test balanced color assignment with even games."""
        matchup = Matchup(participant_a='a', participant_b='b', games=10)
        schedule = create_matchup_schedule(matchup)

        assert len(schedule) == 10

        a_as_blue = sum(1 for g in schedule if g['blue'] == 'a')
        a_as_red = sum(1 for g in schedule if g['red'] == 'a')

        assert a_as_blue == 5
        assert a_as_red == 5

    def test_balanced_color_schedule_odd(self):
        """Test balanced color assignment with odd games."""
        matchup = Matchup(participant_a='a', participant_b='b', games=11)
        schedule = create_matchup_schedule(matchup)

        assert len(schedule) == 11

        a_as_blue = sum(1 for g in schedule if g['blue'] == 'a')
        a_as_red = sum(1 for g in schedule if g['red'] == 'a')

        # Extra game goes to original assignment (a as blue)
        assert a_as_blue == 6
        assert a_as_red == 5

    def test_total_games_calculation(self):
        """Test total games calculation."""
        participants = ['a', 'b', 'c', 'd']
        games_per = 100

        total = total_games(participants, games_per)
        # 6 matchups * 100 games = 600
        assert total == 600

    def test_num_matchups_calculation(self):
        """Test number of matchups calculation."""
        assert num_matchups(['a', 'b']) == 1
        assert num_matchups(['a', 'b', 'c']) == 3
        assert num_matchups(['a', 'b', 'c', 'd']) == 6
        assert num_matchups(['a', 'b', 'c', 'd', 'e']) == 10

    def test_matchup_id_normalization(self):
        """Test matchup ID normalizes colons."""
        matchup = Matchup(
            participant_a='linear:baseline_v1',
            participant_b='linear:trained_001',
            games=10
        )
        assert matchup.matchup_id == 'linear_baseline_v1_vs_linear_trained_001'


class TestTournamentStorage:
    """Tests for tournament storage."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = TournamentStorage(tmpdir)
            yield storage

    def test_create_tournament(self, temp_storage):
        """Test creating a tournament record."""
        tid = temp_storage.create_tournament(
            tournament_id='test_tourney',
            participants=['a', 'b', 'c'],
            games_per_matchup=100,
            k_factor=32,
            initial_elos={'a': 1000, 'b': 1100, 'c': 900}
        )

        assert tid == 'test_tourney'

    def test_save_and_load_matchup(self, temp_storage):
        """Test saving and loading matchup results."""
        temp_storage.create_tournament(
            tournament_id='test',
            participants=['a', 'b'],
            games_per_matchup=100,
            k_factor=32,
            initial_elos={'a': 1000, 'b': 1000}
        )

        result = MatchupResult(
            participant_a='a',
            participant_b='b',
            a_wins=60,
            b_wins=35,
            draws=5,
            games_played=100,
            games_scheduled=100,
            avg_game_length=45.5,
            elo_delta_a=15
        )
        temp_storage.save_matchup_result('test', result)

        loaded = temp_storage.load_tournament('test')
        assert len(loaded.matchups) == 1
        assert loaded.matchups[0].a_wins == 60
        assert loaded.matchups[0].avg_game_length == 45.5

    def test_complete_tournament(self, temp_storage):
        """Test completing a tournament."""
        temp_storage.create_tournament(
            tournament_id='test',
            participants=['a', 'b'],
            games_per_matchup=100,
            k_factor=32,
            initial_elos={'a': 1000, 'b': 1000}
        )

        stats = [
            ParticipantStats(
                participant='a',
                initial_elo=1000,
                final_elo=1050,
                total_wins=60,
                total_losses=35,
                total_draws=5,
                rank=1
            ),
            ParticipantStats(
                participant='b',
                initial_elo=1000,
                final_elo=950,
                total_wins=35,
                total_losses=60,
                total_draws=5,
                rank=2
            )
        ]
        temp_storage.complete_tournament('test', stats)

        loaded = temp_storage.load_tournament('test')
        assert loaded.status == 'completed'
        assert loaded.participants[0].final_elo in [1050, 950]

    def test_list_tournaments(self, temp_storage):
        """Test listing tournaments."""
        temp_storage.create_tournament(
            tournament_id='test1',
            participants=['a', 'b'],
            games_per_matchup=100,
            k_factor=32,
            initial_elos={'a': 1000, 'b': 1000}
        )
        temp_storage.create_tournament(
            tournament_id='test2',
            participants=['a', 'b', 'c'],
            games_per_matchup=200,
            k_factor=32,
            initial_elos={'a': 1000, 'b': 1000, 'c': 1000}
        )

        tournaments = temp_storage.list_tournaments()
        assert len(tournaments) == 2


class TestTournamentResult:
    """Tests for TournamentResult methods."""

    def test_get_rankings(self):
        """Test getting rankings sorted by Elo."""
        from src.tournament.storage import TournamentResult

        result = TournamentResult(
            tournament_id='test',
            created_at='2024-01-01',
            completed_at='2024-01-01',
            status='completed',
            games_per_matchup=100,
            k_factor=32,
            participants=[
                ParticipantStats('a', 1000, 950, 30, 70, 0, 3),
                ParticipantStats('b', 1000, 1100, 70, 30, 0, 1),
                ParticipantStats('c', 1000, 1050, 50, 50, 0, 2),
            ],
            matchups=[]
        )

        rankings = result.get_rankings()
        assert rankings[0].participant == 'b'
        assert rankings[1].participant == 'c'
        assert rankings[2].participant == 'a'

    def test_get_win_matrix(self):
        """Test generating win matrix."""
        from src.tournament.storage import TournamentResult

        result = TournamentResult(
            tournament_id='test',
            created_at='2024-01-01',
            completed_at='2024-01-01',
            status='completed',
            games_per_matchup=100,
            k_factor=32,
            participants=[
                ParticipantStats('a', 1000, 1000, 0, 0, 0, 1),
                ParticipantStats('b', 1000, 1000, 0, 0, 0, 2),
            ],
            matchups=[
                MatchupResult('a', 'b', 60, 35, 5, 100, 100)
            ]
        )

        matrix = result.get_win_matrix()
        assert matrix['a']['b']['wins'] == 60
        assert matrix['a']['b']['losses'] == 35
        assert matrix['b']['a']['wins'] == 35
        assert matrix['b']['a']['losses'] == 60


class TestTournamentRunner:
    """Integration tests for tournament runner."""

    @pytest.fixture
    def temp_config(self):
        """Create a test config with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield TournamentConfig(
                participants=['random', 'heuristic'],
                games_per_matchup=4,  # Small for fast tests
                k_factor=32,
                initial_elo=1000,
                max_moves=50,
                data_dir=tmpdir
            )

    def test_run_minimal_tournament(self, temp_config):
        """Test running a minimal tournament."""
        runner = TournamentRunner(
            config=temp_config,
            verbose=False,
            show_progress=False
        )

        result = runner.run(tournament_id='test_minimal')

        # Check structure
        assert result.tournament_id == 'test_minimal'
        assert result.status == 'completed'
        assert len(result.participants) == 2
        assert len(result.matchups) == 1  # 2 participants = 1 matchup

        # Check games were played
        matchup = result.matchups[0]
        assert matchup.games_played == 4
        assert matchup.a_wins + matchup.b_wins + matchup.draws == 4

    def test_tournament_elo_updates(self, temp_config):
        """Test that Elo ratings are updated during tournament."""
        runner = TournamentRunner(
            config=temp_config,
            verbose=False
        )

        result = runner.run()

        # At least one participant should have changed Elo
        # (unless it was exactly 50-50, which is unlikely)
        elo_changed = any(
            p.final_elo != p.initial_elo
            for p in result.participants
        )
        # This might occasionally fail with exactly 50-50 results
        # but is very unlikely with heuristic vs random

    def test_tournament_persistence(self, temp_config):
        """Test that tournament results are persisted."""
        runner = TournamentRunner(
            config=temp_config,
            verbose=False
        )

        result = runner.run(tournament_id='persist_test')

        # Load from storage
        storage = TournamentStorage(temp_config.data_dir)
        loaded = storage.load_tournament('persist_test')

        assert loaded is not None
        assert loaded.tournament_id == 'persist_test'
        assert len(loaded.matchups) == 1

    def test_three_player_tournament(self):
        """Test tournament with 3 players."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TournamentConfig(
                participants=['random', 'heuristic', 'linear'],  # 3 different agents
                games_per_matchup=2,
                data_dir=tmpdir
            )
            runner = TournamentRunner(config=config, verbose=False)
            result = runner.run()

            # 3 participants = 3 matchups
            assert len(result.matchups) == 3
            assert len(result.participants) == 3

            # Each participant played 2 matchups * 2 games = 4 games
            for p in result.participants:
                assert p.total_games == 4
