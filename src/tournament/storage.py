"""
Storage backend for tournament results.

Uses SQLite for tournament metadata, participants, and matchup results.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class MatchupResult:
    """Results for a single pairwise matchup."""
    participant_a: str
    participant_b: str
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    games_played: int = 0
    games_scheduled: int = 0
    avg_game_length: float = 0.0
    elo_delta_a: int = 0

    @property
    def matchup_id(self) -> str:
        a_norm = self.participant_a.replace(':', '_')
        b_norm = self.participant_b.replace(':', '_')
        return f"{a_norm}_vs_{b_norm}"

    @property
    def a_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.a_wins / self.games_played


@dataclass
class ParticipantStats:
    """Aggregate stats for a tournament participant."""
    participant: str
    initial_elo: int
    final_elo: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_draws: int = 0
    rank: int = 0

    @property
    def total_games(self) -> int:
        return self.total_wins + self.total_losses + self.total_draws

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.total_wins / self.total_games


@dataclass
class TournamentResult:
    """Complete results of a tournament."""
    tournament_id: str
    created_at: str
    completed_at: Optional[str]
    status: str
    games_per_matchup: int
    k_factor: int
    participants: List[ParticipantStats]
    matchups: List[MatchupResult]

    def get_rankings(self) -> List[ParticipantStats]:
        """Return participants sorted by final Elo (descending)."""
        return sorted(self.participants, key=lambda p: p.final_elo, reverse=True)

    def get_win_matrix(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Generate a win matrix from matchup results.

        Returns:
            {
                'agent_a': {
                    'agent_b': {'wins': X, 'losses': Y, 'draws': Z},
                    ...
                },
                ...
            }
        """
        participant_names = [p.participant for p in self.participants]
        matrix = {p: {} for p in participant_names}

        for matchup in self.matchups:
            a, b = matchup.participant_a, matchup.participant_b
            matrix[a][b] = {
                'wins': matchup.a_wins,
                'losses': matchup.b_wins,
                'draws': matchup.draws
            }
            matrix[b][a] = {
                'wins': matchup.b_wins,
                'losses': matchup.a_wins,
                'draws': matchup.draws
            }

        return matrix


class TournamentStorage:
    """
    Handles persistent storage of tournament results.

    Uses SQLite tables in the games.db database.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize storage backend.

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "games.db"

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database tables
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database schema for tournaments."""
        with sqlite3.connect(self.db_path) as conn:
            # Tournament metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tournaments (
                    tournament_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT DEFAULT 'in_progress',
                    games_per_matchup INTEGER NOT NULL,
                    k_factor INTEGER DEFAULT 32,
                    config TEXT
                )
            """)

            # Participant final standings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tournament_participants (
                    tournament_id TEXT NOT NULL,
                    participant TEXT NOT NULL,
                    initial_elo INTEGER,
                    final_elo INTEGER,
                    total_wins INTEGER DEFAULT 0,
                    total_losses INTEGER DEFAULT 0,
                    total_draws INTEGER DEFAULT 0,
                    rank INTEGER,
                    PRIMARY KEY (tournament_id, participant),
                    FOREIGN KEY (tournament_id) REFERENCES tournaments(tournament_id)
                )
            """)

            # Matchup results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tournament_matchups (
                    tournament_id TEXT NOT NULL,
                    matchup_id TEXT NOT NULL,
                    participant_a TEXT NOT NULL,
                    participant_b TEXT NOT NULL,
                    a_wins INTEGER DEFAULT 0,
                    b_wins INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    games_played INTEGER DEFAULT 0,
                    games_scheduled INTEGER NOT NULL,
                    avg_game_length REAL,
                    elo_delta_a INTEGER,
                    PRIMARY KEY (tournament_id, matchup_id),
                    FOREIGN KEY (tournament_id) REFERENCES tournaments(tournament_id)
                )
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tournaments_status ON tournaments(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_participants_tournament ON tournament_participants(tournament_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_matchups_tournament ON tournament_matchups(tournament_id)")

            conn.commit()

    def create_tournament(
        self,
        tournament_id: str,
        participants: List[str],
        games_per_matchup: int,
        k_factor: int,
        initial_elos: Dict[str, int],
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new tournament record.

        Args:
            tournament_id: Unique tournament identifier
            participants: List of participant agent specs
            games_per_matchup: Number of games per matchup
            k_factor: Elo K-factor
            initial_elos: Initial Elo ratings for each participant
            config: Optional configuration dict

        Returns:
            tournament_id
        """
        created_at = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Insert tournament
            conn.execute("""
                INSERT INTO tournaments (tournament_id, created_at, status, games_per_matchup, k_factor, config)
                VALUES (?, ?, 'in_progress', ?, ?, ?)
            """, (tournament_id, created_at, games_per_matchup, k_factor, json.dumps(config or {})))

            # Insert participants
            for p in participants:
                conn.execute("""
                    INSERT INTO tournament_participants (tournament_id, participant, initial_elo)
                    VALUES (?, ?, ?)
                """, (tournament_id, p, initial_elos.get(p, 1000)))

            conn.commit()

        return tournament_id

    def save_matchup_result(self, tournament_id: str, result: MatchupResult):
        """Save or update a matchup result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tournament_matchups
                (tournament_id, matchup_id, participant_a, participant_b,
                 a_wins, b_wins, draws, games_played, games_scheduled,
                 avg_game_length, elo_delta_a)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tournament_id,
                result.matchup_id,
                result.participant_a,
                result.participant_b,
                result.a_wins,
                result.b_wins,
                result.draws,
                result.games_played,
                result.games_scheduled,
                result.avg_game_length,
                result.elo_delta_a
            ))
            conn.commit()

    def complete_tournament(
        self,
        tournament_id: str,
        participant_stats: List[ParticipantStats]
    ):
        """Mark tournament as completed and save final standings."""
        completed_at = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Update tournament status
            conn.execute("""
                UPDATE tournaments
                SET status = 'completed', completed_at = ?
                WHERE tournament_id = ?
            """, (completed_at, tournament_id))

            # Update participant stats
            for stats in participant_stats:
                conn.execute("""
                    UPDATE tournament_participants
                    SET final_elo = ?, total_wins = ?, total_losses = ?,
                        total_draws = ?, rank = ?
                    WHERE tournament_id = ? AND participant = ?
                """, (
                    stats.final_elo,
                    stats.total_wins,
                    stats.total_losses,
                    stats.total_draws,
                    stats.rank,
                    tournament_id,
                    stats.participant
                ))

            conn.commit()

    def load_tournament(self, tournament_id: str) -> Optional[TournamentResult]:
        """Load a tournament by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get tournament
            cursor = conn.execute(
                "SELECT * FROM tournaments WHERE tournament_id = ?",
                (tournament_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Get participants
            cursor = conn.execute(
                "SELECT * FROM tournament_participants WHERE tournament_id = ?",
                (tournament_id,)
            )
            participants = [
                ParticipantStats(
                    participant=r['participant'],
                    initial_elo=r['initial_elo'] or 1000,
                    final_elo=r['final_elo'] or r['initial_elo'] or 1000,
                    total_wins=r['total_wins'] or 0,
                    total_losses=r['total_losses'] or 0,
                    total_draws=r['total_draws'] or 0,
                    rank=r['rank'] or 0
                )
                for r in cursor.fetchall()
            ]

            # Get matchups
            cursor = conn.execute(
                "SELECT * FROM tournament_matchups WHERE tournament_id = ?",
                (tournament_id,)
            )
            matchups = [
                MatchupResult(
                    participant_a=r['participant_a'],
                    participant_b=r['participant_b'],
                    a_wins=r['a_wins'] or 0,
                    b_wins=r['b_wins'] or 0,
                    draws=r['draws'] or 0,
                    games_played=r['games_played'] or 0,
                    games_scheduled=r['games_scheduled'] or 0,
                    avg_game_length=r['avg_game_length'] or 0.0,
                    elo_delta_a=r['elo_delta_a'] or 0
                )
                for r in cursor.fetchall()
            ]

            return TournamentResult(
                tournament_id=row['tournament_id'],
                created_at=row['created_at'],
                completed_at=row['completed_at'],
                status=row['status'],
                games_per_matchup=row['games_per_matchup'],
                k_factor=row['k_factor'] or 32,
                participants=participants,
                matchups=matchups
            )

    def list_tournaments(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent tournaments."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT tournament_id, created_at, completed_at, status, games_per_matchup
                FROM tournaments
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]
