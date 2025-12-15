"""
Storage backend for game trajectories.

Uses JSONL files for trajectory data and SQLite for indexing/querying.
"""
import json
import sqlite3
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from src.logging.trajectory import GameTrajectory


class GameStorage:
    """
    Handles persistent storage of game trajectories.

    - JSONL files: Store complete trajectories (one JSON object per line)
    - SQLite: Index games for fast querying by metadata
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize storage backend.

        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.games_dir = self.data_dir / "games"
        self.db_path = self.data_dir / "games.db"

        # Ensure directories exist
        self.games_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    blue_agent TEXT NOT NULL,
                    red_agent TEXT NOT NULL,
                    winner INTEGER,
                    total_moves INTEGER NOT NULL,
                    cards_used TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_number INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_games_timestamp ON games(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_games_agents ON games(blue_agent, red_agent)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_games_winner ON games(winner)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_games_moves ON games(total_moves)")
            conn.commit()

    def _get_jsonl_path(self, date: Optional[datetime] = None) -> Path:
        """Get the JSONL file path for a given date."""
        if date is None:
            date = datetime.utcnow()
        filename = f"games_{date.strftime('%Y%m%d')}.jsonl"
        return self.games_dir / filename

    def _count_lines(self, file_path: Path) -> int:
        """Count number of lines in a file."""
        if not file_path.exists():
            return 0
        with open(file_path, 'r') as f:
            return sum(1 for _ in f)

    def save_trajectory(self, trajectory: GameTrajectory) -> str:
        """
        Save a game trajectory to storage.

        Args:
            trajectory: Complete game trajectory to save

        Returns:
            game_id of saved trajectory
        """
        # Get file path and line number
        jsonl_path = self._get_jsonl_path()
        line_number = self._count_lines(jsonl_path)

        # Write to JSONL file
        with open(jsonl_path, 'a') as f:
            json.dump(trajectory.to_dict(), f, separators=(',', ':'))
            f.write('\n')

        # Index in SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO games
                (game_id, timestamp, blue_agent, red_agent, winner,
                 total_moves, cards_used, file_path, line_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trajectory.game_id,
                trajectory.timestamp,
                trajectory.config.blue_agent if trajectory.config else "unknown",
                trajectory.config.red_agent if trajectory.config else "unknown",
                trajectory.outcome.winner if trajectory.outcome else None,
                trajectory.outcome.total_moves if trajectory.outcome else 0,
                json.dumps(trajectory.config.cards_used if trajectory.config else []),
                str(jsonl_path.relative_to(self.data_dir)),
                line_number
            ))
            conn.commit()

        return trajectory.game_id

    def load_trajectory(self, game_id: str) -> Optional[GameTrajectory]:
        """
        Load a game trajectory by ID.

        Args:
            game_id: UUID of the game

        Returns:
            GameTrajectory or None if not found
        """
        # Look up file location in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_path, line_number FROM games WHERE game_id = ?",
                (game_id,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        file_path, line_number = row
        full_path = self.data_dir / file_path

        if not full_path.exists():
            return None

        # Read the specific line
        with open(full_path, 'r') as f:
            for i, line in enumerate(f):
                if i == line_number:
                    data = json.loads(line)
                    return GameTrajectory.from_dict(data)

        return None

    def query_games(
        self,
        blue_agent: Optional[str] = None,
        red_agent: Optional[str] = None,
        winner: Optional[int] = None,
        min_moves: Optional[int] = None,
        max_moves: Optional[int] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        agent_match_mode: str = "contains"
    ) -> List[Dict[str, Any]]:
        """
        Query games by various criteria.

        Args:
            blue_agent: Filter by blue agent type
            red_agent: Filter by red agent type
            winner: Filter by winner (0=BLUE, 1=RED, None for all including draws)
            min_moves: Minimum number of moves
            max_moves: Maximum number of moves
            since: ISO timestamp - games after this time
            until: ISO timestamp - games before this time
            limit: Maximum results to return
            offset: Number of results to skip
            agent_match_mode: How to match agent names:
                - "exact": Exact string match
                - "contains": Substring match (default)
                - "prefix": Match agent type prefix (e.g., "linear:" matches "linear:model_name")

        Returns:
            List of game summary dictionaries
        """
        conditions = []
        params = []

        if blue_agent:
            if agent_match_mode == "exact":
                conditions.append("blue_agent = ?")
                params.append(blue_agent)
            elif agent_match_mode == "contains":
                conditions.append("blue_agent LIKE ?")
                params.append(f"%{blue_agent}%")
            elif agent_match_mode == "prefix":
                conditions.append("blue_agent LIKE ?")
                params.append(f"{blue_agent}:%")
        if red_agent:
            if agent_match_mode == "exact":
                conditions.append("red_agent = ?")
                params.append(red_agent)
            elif agent_match_mode == "contains":
                conditions.append("red_agent LIKE ?")
                params.append(f"%{red_agent}%")
            elif agent_match_mode == "prefix":
                conditions.append("red_agent LIKE ?")
                params.append(f"{red_agent}:%")
        if winner is not None:
            conditions.append("winner = ?")
            params.append(winner)
        if min_moves is not None:
            conditions.append("total_moves >= ?")
            params.append(min_moves)
        if max_moves is not None:
            conditions.append("total_moves <= ?")
            params.append(max_moves)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT game_id, timestamp, blue_agent, red_agent, winner,
                   total_moves, cards_used
            FROM games
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [
            {
                "game_id": row["game_id"],
                "timestamp": row["timestamp"],
                "blue_agent": row["blue_agent"],
                "red_agent": row["red_agent"],
                "winner": row["winner"],
                "total_moves": row["total_moves"],
                "cards_used": json.loads(row["cards_used"])
            }
            for row in rows
        ]

    def count_games(
        self,
        blue_agent: Optional[str] = None,
        red_agent: Optional[str] = None,
        winner: Optional[int] = None
    ) -> int:
        """Count games matching criteria."""
        conditions = []
        params = []

        if blue_agent:
            conditions.append("blue_agent = ?")
            params.append(blue_agent)
        if red_agent:
            conditions.append("red_agent = ?")
            params.append(red_agent)
        if winner is not None:
            conditions.append("winner = ?")
            params.append(winner)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM games WHERE {where_clause}",
                params
            )
            return cursor.fetchone()[0]

    def get_unique_agent_combinations(self) -> List[Tuple[str, str, int]]:
        """
        Get all unique agent matchups with counts.

        Returns:
            List of tuples: (blue_agent, red_agent, count)
            Sorted by count descending
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT blue_agent, red_agent, COUNT(*) as count
                FROM games
                GROUP BY blue_agent, red_agent
                ORDER BY count DESC
            """)
            return cursor.fetchall()

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics about stored games."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            blue_wins = conn.execute("SELECT COUNT(*) FROM games WHERE winner = 0").fetchone()[0]
            red_wins = conn.execute("SELECT COUNT(*) FROM games WHERE winner = 1").fetchone()[0]
            draws = conn.execute("SELECT COUNT(*) FROM games WHERE winner IS NULL").fetchone()[0]
            avg_moves = conn.execute("SELECT AVG(total_moves) FROM games").fetchone()[0] or 0

        return {
            "total_games": total,
            "blue_wins": blue_wins,
            "red_wins": red_wins,
            "draws": draws,
            "average_moves": round(avg_moves, 1)
        }
