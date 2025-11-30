"""
Display formatting for tournament results.

Provides ASCII-formatted leaderboards and win matrices for terminal output.
"""

from typing import List, Dict
from src.tournament.storage import TournamentResult, ParticipantStats


def format_leaderboard(result: TournamentResult) -> str:
    """
    Format the final standings as an ASCII table.

    Args:
        result: Complete tournament result

    Returns:
        Formatted string for terminal display
    """
    rankings = result.get_rankings()

    lines = []
    lines.append("=== FINAL STANDINGS ===")
    lines.append("")

    # Header
    lines.append(f"{'Rank':<6}{'Participant':<28}{'Elo':<8}{'W-L-D':<14}{'Win%':<8}")
    lines.append("-" * 64)

    # Rows
    for i, stats in enumerate(rankings, 1):
        wld = f"{stats.total_wins}-{stats.total_losses}-{stats.total_draws}"
        win_pct = f"{stats.win_rate:.1%}"
        elo_change = stats.final_elo - stats.initial_elo
        elo_str = f"{stats.final_elo}"
        if elo_change != 0:
            sign = "+" if elo_change > 0 else ""
            elo_str = f"{stats.final_elo} ({sign}{elo_change})"

        lines.append(f"{i:<6}{stats.participant:<28}{elo_str:<8}{wld:<14}{win_pct:<8}")

    return "\n".join(lines)


def format_win_matrix(result: TournamentResult) -> str:
    """
    Format the win matrix as an ASCII table.

    Shows wins-losses for row participant vs column participant.

    Args:
        result: Complete tournament result

    Returns:
        Formatted string for terminal display
    """
    matrix = result.get_win_matrix()
    participants = [p.participant for p in result.get_rankings()]

    # Truncate long names for display
    def short_name(name: str, max_len: int = 14) -> str:
        if len(name) <= max_len:
            return name
        return name[:max_len-2] + ".."

    short_names = [short_name(p) for p in participants]

    lines = []
    lines.append("")
    lines.append("Win Matrix (row W-L vs column):")
    lines.append("")

    # Calculate column width
    col_width = max(len(sn) for sn in short_names) + 2
    col_width = max(col_width, 10)

    # Header row
    header = " " * (col_width + 2)
    for sn in short_names:
        header += f"{sn:>{col_width}}"
    lines.append(header)

    # Data rows
    for i, (p, sn) in enumerate(zip(participants, short_names)):
        row = f"{sn:<{col_width}}  "
        for j, op in enumerate(participants):
            if i == j:
                cell = "-"
            else:
                record = matrix[p].get(op, {'wins': 0, 'losses': 0})
                cell = f"{record['wins']}-{record['losses']}"
            row += f"{cell:>{col_width}}"
        lines.append(row)

    return "\n".join(lines)


def format_matchup_result(
    matchup_num: int,
    total_matchups: int,
    participant_a: str,
    participant_b: str,
    a_wins: int,
    b_wins: int,
    draws: int
) -> str:
    """Format a single matchup result line."""
    return (f"[{matchup_num}/{total_matchups}] "
            f"{participant_a} vs {participant_b}: "
            f"{a_wins}W-{b_wins}L-{draws}D")


def format_progress(
    games_completed: int,
    total_games: int,
    games_per_second: float,
    eta_seconds: float
) -> str:
    """Format progress line during tournament execution."""
    pct = games_completed / total_games if total_games > 0 else 0
    return (f"  Progress: {games_completed}/{total_games} ({pct:.1%}), "
            f"Rate: {games_per_second:.1f} games/s, "
            f"ETA: {eta_seconds:.0f}s")


def format_tournament_header(
    tournament_id: str,
    num_participants: int,
    games_per_matchup: int,
    total_games: int
) -> str:
    """Format tournament header information."""
    lines = []
    lines.append(f"Tournament: {tournament_id}")
    lines.append(f"Participants: {num_participants}")
    lines.append(f"Games per matchup: {games_per_matchup}")
    lines.append(f"Total games: {total_games}")
    lines.append("")
    return "\n".join(lines)
