"""
Elo rating calculator for tournament rankings.

Implements the standard Elo rating system:
- Expected score: E = 1 / (1 + 10^((R_opp - R_self) / 400))
- Rating update: R_new = R_old + K * (S - E)
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class EloUpdate:
    """Result of an Elo update after a matchup."""
    participant_a: str
    participant_b: str
    old_elo_a: int
    old_elo_b: int
    new_elo_a: int
    new_elo_b: int
    expected_a: float
    actual_score_a: float
    delta_a: int


class EloCalculator:
    """
    Standard Elo rating calculator.

    Updates ratings after each matchup based on aggregate win rate.
    """

    # Default Elo for built-in agents
    BUILTIN_DEFAULTS = {
        'random': 800,
        'heuristic': 1000,
        'linear': 1000,
    }

    def __init__(self, k_factor: int = 32, initial_elo: int = 1000):
        """
        Initialize the Elo calculator.

        Args:
            k_factor: The K-factor determines rating volatility (default: 32)
            initial_elo: Default starting Elo for new participants
        """
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.ratings: Dict[str, int] = {}

    def get_rating(self, participant: str) -> int:
        """Get current rating for a participant."""
        return self.ratings.get(participant, self.initial_elo)

    def set_initial_ratings(self, participants: list, existing_elos: Optional[Dict[str, int]] = None):
        """
        Initialize ratings for all participants.

        Args:
            participants: List of participant agent specs
            existing_elos: Optional dict of pre-existing Elo ratings
        """
        existing_elos = existing_elos or {}
        for p in participants:
            if p in existing_elos:
                self.ratings[p] = existing_elos[p]
            else:
                # Check if it's a built-in agent type
                agent_type = p.split(':')[0] if ':' in p else p
                self.ratings[p] = self.BUILTIN_DEFAULTS.get(agent_type, self.initial_elo)

    def expected_score(self, rating_a: int, rating_b: int) -> float:
        """
        Calculate expected score for player A against player B.

        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B

        Returns:
            Expected score between 0 and 1
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_from_matchup(
        self,
        participant_a: str,
        participant_b: str,
        a_wins: int,
        b_wins: int,
        draws: int
    ) -> EloUpdate:
        """
        Update ratings after a completed matchup.

        Uses aggregate score from all games in the matchup.

        Args:
            participant_a: First participant
            participant_b: Second participant
            a_wins: Number of wins for A
            b_wins: Number of wins for B
            draws: Number of draws

        Returns:
            EloUpdate with old/new ratings and calculation details
        """
        total_games = a_wins + b_wins + draws
        if total_games == 0:
            raise ValueError("Cannot update Elo with zero games")

        # Calculate aggregate score (wins = 1, draws = 0.5, losses = 0)
        actual_score_a = (a_wins + 0.5 * draws) / total_games

        # Get current ratings
        old_elo_a = self.get_rating(participant_a)
        old_elo_b = self.get_rating(participant_b)

        # Calculate expected score
        expected_a = self.expected_score(old_elo_a, old_elo_b)

        # Calculate rating change
        # Scale K-factor by sqrt(games) to give more weight to larger matchups
        # but cap the scaling to avoid extreme swings
        effective_k = self.k_factor * min(total_games ** 0.5, 10) / 3.16  # sqrt(10) ≈ 3.16
        delta_a = round(effective_k * (actual_score_a - expected_a))

        # Update ratings
        new_elo_a = old_elo_a + delta_a
        new_elo_b = old_elo_b - delta_a

        self.ratings[participant_a] = new_elo_a
        self.ratings[participant_b] = new_elo_b

        return EloUpdate(
            participant_a=participant_a,
            participant_b=participant_b,
            old_elo_a=old_elo_a,
            old_elo_b=old_elo_b,
            new_elo_a=new_elo_a,
            new_elo_b=new_elo_b,
            expected_a=expected_a,
            actual_score_a=actual_score_a,
            delta_a=delta_a
        )

    def get_all_ratings(self) -> Dict[str, int]:
        """Get a copy of all current ratings."""
        return self.ratings.copy()
