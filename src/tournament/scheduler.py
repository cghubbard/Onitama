"""
Round-robin tournament scheduling.

Generates matchups between all participants and balanced color assignments.
"""

from itertools import combinations
from dataclasses import dataclass
from typing import List
import random


@dataclass
class Matchup:
    """Represents a single matchup between two participants."""
    participant_a: str
    participant_b: str
    games: int

    @property
    def matchup_id(self) -> str:
        """Generate a unique ID for this matchup."""
        a_norm = self.participant_a.replace(':', '_')
        b_norm = self.participant_b.replace(':', '_')
        return f"{a_norm}_vs_{b_norm}"


def generate_round_robin_schedule(
    participants: List[str],
    games_per_matchup: int,
    shuffle: bool = True
) -> List[Matchup]:
    """
    Generate all matchups for a round-robin tournament.

    Args:
        participants: List of participant agent specs
        games_per_matchup: Number of games per matchup
        shuffle: Whether to shuffle matchup order (default: True)

    Returns:
        List of Matchup objects

    Raises:
        ValueError: If fewer than 2 participants
    """
    if len(participants) < 2:
        raise ValueError("Need at least 2 participants for a tournament")

    matchups = []
    for a, b in combinations(participants, 2):
        matchups.append(Matchup(
            participant_a=a,
            participant_b=b,
            games=games_per_matchup
        ))

    if shuffle:
        random.shuffle(matchups)

    return matchups


def create_matchup_schedule(matchup: Matchup) -> List[dict]:
    """
    Create balanced color schedule for a single matchup.

    First half: A as BLUE, B as RED
    Second half: B as BLUE, A as RED
    If odd games, extra game goes to original assignment.

    Args:
        matchup: The Matchup to create schedule for

    Returns:
        List of dicts with 'blue' and 'red' keys
    """
    schedule = []
    games_per_side = matchup.games // 2

    # First half: A plays BLUE
    for _ in range(games_per_side):
        schedule.append({
            'blue': matchup.participant_a,
            'red': matchup.participant_b
        })

    # Second half: A plays RED
    for _ in range(games_per_side):
        schedule.append({
            'blue': matchup.participant_b,
            'red': matchup.participant_a
        })

    # Handle odd number of games
    if matchup.games % 2 == 1:
        schedule.append({
            'blue': matchup.participant_a,
            'red': matchup.participant_b
        })

    return schedule


def total_games(participants: List[str], games_per_matchup: int) -> int:
    """Calculate total games in a round-robin tournament."""
    n = len(participants)
    num_matchups = n * (n - 1) // 2
    return num_matchups * games_per_matchup


def num_matchups(participants: List[str]) -> int:
    """Calculate number of matchups in a round-robin tournament."""
    n = len(participants)
    return n * (n - 1) // 2
