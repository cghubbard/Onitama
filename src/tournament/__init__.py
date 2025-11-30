"""
Tournament module for running round-robin competitions between agents.

Provides:
- EloCalculator: Standard Elo rating calculation
- TournamentRunner: Orchestrates tournament execution
- TournamentStorage: Persists tournament results
"""

from src.tournament.elo import EloCalculator
from src.tournament.scheduler import generate_round_robin_schedule, create_matchup_schedule, Matchup
from src.tournament.storage import TournamentStorage
from src.tournament.runner import TournamentRunner
from src.tournament.display import format_leaderboard, format_win_matrix

__all__ = [
    'EloCalculator',
    'TournamentRunner',
    'TournamentStorage',
    'generate_round_robin_schedule',
    'create_matchup_schedule',
    'Matchup',
    'format_leaderboard',
    'format_win_matrix',
]
