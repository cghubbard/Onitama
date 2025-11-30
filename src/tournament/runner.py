"""
Tournament runner that orchestrates round-robin competitions.

Handles game execution, result tracking, Elo updates, and progress reporting.
"""

import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass

from src.game.game import Game
from src.utils.constants import BLUE, RED, ONGOING, DRAW
from src.tournament.elo import EloCalculator
from src.tournament.scheduler import (
    generate_round_robin_schedule,
    create_matchup_schedule,
    Matchup,
    total_games,
    num_matchups
)
from src.tournament.storage import (
    TournamentStorage,
    TournamentResult,
    MatchupResult,
    ParticipantStats
)
from src.tournament.display import (
    format_tournament_header,
    format_matchup_result,
    format_progress
)

# Import agent creation from main
from main import create_agent, parse_agent_spec, AGENT_TYPES

# Import game logging
from src.logging.game_logger import GameLogger, create_logger_from_args


@dataclass
class TournamentConfig:
    """Configuration for a tournament run."""
    participants: List[str]
    games_per_matchup: int = 500
    k_factor: int = 32
    initial_elo: int = 1000
    max_moves: int = 200
    log_games: bool = False
    data_dir: str = "data"


class TournamentRunner:
    """
    Orchestrates a round-robin tournament between multiple agents.

    Usage:
        runner = TournamentRunner(config)
        result = runner.run(tournament_id="my_tournament")
    """

    def __init__(
        self,
        config: TournamentConfig,
        storage: Optional[TournamentStorage] = None,
        verbose: bool = True,
        show_progress: bool = False
    ):
        """
        Initialize the tournament runner.

        Args:
            config: Tournament configuration
            storage: Optional storage backend (creates default if None)
            verbose: Print matchup results
            show_progress: Show progress updates during matchups
        """
        self.config = config
        self.storage = storage or TournamentStorage(config.data_dir)
        self.verbose = verbose
        self.show_progress = show_progress

        # Initialize Elo calculator
        self.elo = EloCalculator(
            k_factor=config.k_factor,
            initial_elo=config.initial_elo
        )

        # Game logger (if logging enabled)
        self.logger = None
        if config.log_games:
            self.logger = create_logger_from_args('all', 1.0, config.data_dir)

        # Validate participants
        self._validate_participants()

    def _validate_participants(self):
        """Validate that all participant agent specs are valid."""
        for spec in self.config.participants:
            agent_type, model_name = parse_agent_spec(spec)
            if agent_type not in AGENT_TYPES:
                raise ValueError(f"Unknown agent type: {agent_type}")
            # Try creating an agent to validate model exists
            try:
                create_agent(spec, BLUE)
            except Exception as e:
                raise ValueError(f"Invalid agent spec '{spec}': {e}")

    def _get_initial_elos(self) -> Dict[str, int]:
        """Get initial Elo ratings for all participants."""
        from src.evaluation.model_store import ModelStore

        elos = {}
        store = ModelStore()

        for p in self.config.participants:
            agent_type, model_name = parse_agent_spec(p)

            # Try to load from ModelStore for model-based agents
            if model_name and store.exists(model_name):
                model = store.load(model_name)
                if model.elo:
                    elos[p] = model.elo
                    continue

            # Use built-in defaults or initial_elo
            elos[p] = EloCalculator.BUILTIN_DEFAULTS.get(
                agent_type, self.config.initial_elo
            )

        return elos

    def _generate_tournament_id(self) -> str:
        """Generate a unique tournament ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"tourney_{timestamp}"

    def run(self, tournament_id: Optional[str] = None) -> TournamentResult:
        """
        Run a complete round-robin tournament.

        Args:
            tournament_id: Optional ID (auto-generated if None)

        Returns:
            Complete TournamentResult
        """
        tournament_id = tournament_id or self._generate_tournament_id()

        # Initialize Elo ratings
        initial_elos = self._get_initial_elos()
        self.elo.set_initial_ratings(self.config.participants, initial_elos)

        # Generate schedule
        matchups = generate_round_robin_schedule(
            self.config.participants,
            self.config.games_per_matchup
        )

        total_games_count = total_games(
            self.config.participants,
            self.config.games_per_matchup
        )

        # Create tournament record
        self.storage.create_tournament(
            tournament_id=tournament_id,
            participants=self.config.participants,
            games_per_matchup=self.config.games_per_matchup,
            k_factor=self.config.k_factor,
            initial_elos=initial_elos
        )

        # Print header
        if self.verbose:
            print(format_tournament_header(
                tournament_id,
                len(self.config.participants),
                self.config.games_per_matchup,
                total_games_count
            ))

        # Run all matchups
        matchup_results = []
        start_time = time.time()
        games_completed = 0

        for i, matchup in enumerate(matchups, 1):
            result = self._run_matchup(
                matchup,
                matchup_num=i,
                total_matchups=len(matchups),
                games_completed_so_far=games_completed,
                total_games=total_games_count,
                start_time=start_time
            )
            matchup_results.append(result)
            games_completed += result.games_played

            # Update Elo after matchup
            elo_update = self.elo.update_from_matchup(
                matchup.participant_a,
                matchup.participant_b,
                result.a_wins,
                result.b_wins,
                result.draws
            )
            result.elo_delta_a = elo_update.delta_a

            # Save matchup result
            self.storage.save_matchup_result(tournament_id, result)

            # Print matchup result
            if self.verbose:
                print(format_matchup_result(
                    i, len(matchups),
                    matchup.participant_a,
                    matchup.participant_b,
                    result.a_wins,
                    result.b_wins,
                    result.draws
                ))

        # Compile final results
        total_time = time.time() - start_time
        if self.verbose:
            print(f"\nTournament completed in {total_time:.1f}s "
                  f"({games_completed} games, {games_completed/total_time:.1f} games/s)")

        # Calculate participant stats
        participant_stats = self._compile_participant_stats(
            matchup_results,
            initial_elos
        )

        # Save final results
        self.storage.complete_tournament(tournament_id, participant_stats)

        return TournamentResult(
            tournament_id=tournament_id,
            created_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            status='completed',
            games_per_matchup=self.config.games_per_matchup,
            k_factor=self.config.k_factor,
            participants=participant_stats,
            matchups=matchup_results
        )

    def _run_matchup(
        self,
        matchup: Matchup,
        matchup_num: int,
        total_matchups: int,
        games_completed_so_far: int,
        total_games: int,
        start_time: float
    ) -> MatchupResult:
        """Run all games for a single matchup."""
        if self.verbose:
            print(f"\n[{matchup_num}/{total_matchups}] "
                  f"{matchup.participant_a} vs {matchup.participant_b}")

        # Create balanced schedule
        schedule = create_matchup_schedule(matchup)

        result = MatchupResult(
            participant_a=matchup.participant_a,
            participant_b=matchup.participant_b,
            games_scheduled=matchup.games
        )

        total_moves = 0
        progress_interval = max(1, matchup.games // 10)

        for game_num, game_config in enumerate(schedule):
            # Run single game
            outcome, moves = self._run_single_game(
                game_config['blue'],
                game_config['red']
            )

            result.games_played += 1
            total_moves += moves

            # Map outcome to participants
            if outcome == 1:  # BLUE_WINS
                if game_config['blue'] == matchup.participant_a:
                    result.a_wins += 1
                else:
                    result.b_wins += 1
            elif outcome == 2:  # RED_WINS
                if game_config['red'] == matchup.participant_a:
                    result.a_wins += 1
                else:
                    result.b_wins += 1
            else:  # DRAW
                result.draws += 1

            # Progress update
            if self.show_progress and (game_num + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                games_now = games_completed_so_far + result.games_played
                rate = games_now / elapsed if elapsed > 0 else 0
                remaining = total_games - games_now
                eta = remaining / rate if rate > 0 else 0
                print(format_progress(games_now, total_games, rate, eta))

        result.avg_game_length = total_moves / result.games_played if result.games_played > 0 else 0
        return result

    def _run_single_game(self, blue_spec: str, red_spec: str) -> tuple:
        """
        Run a single game between two agents.

        Args:
            blue_spec: Agent spec for BLUE player
            red_spec: Agent spec for RED player

        Returns:
            Tuple of (outcome, move_count)
        """
        game = Game()
        blue_agent = create_agent(blue_spec, BLUE)
        red_agent = create_agent(red_spec, RED)
        agents = {BLUE: blue_agent, RED: red_agent}

        # Optional game logging
        log_session = None
        if self.logger:
            log_session = self.logger.start_game(game, blue_spec, red_spec)

        move_count = 0
        history = {}

        while game.get_outcome() == ONGOING and move_count < self.config.max_moves:
            current_player = game.get_current_player()
            agent = agents[current_player]

            move = agent.select_move(game)
            if move is None:
                if log_session:
                    log_session.end_game(None, "no_moves")
                return DRAW, move_count

            # Log pre-move state
            pre_state = None
            if log_session:
                pre_state = log_session.log_pre_move_state(game)

            from_pos, to_pos, card_name = move
            game.make_move(from_pos, to_pos, card_name)
            move_count += 1

            # Log move
            if log_session and pre_state:
                log_session.log_move_with_pre_state(pre_state, move)

            # Cycle detection (same as main.py)
            board_hash = str(sorted(game.get_board_state().items()))
            player_hash = str(game.get_current_player())
            blue_cards = str(sorted(card.name for card in game.get_player_cards(BLUE)))
            red_cards = str(sorted(card.name for card in game.get_player_cards(RED)))
            neutral_card = game.get_neutral_card().name
            state_hash = f"{board_hash}|{player_hash}|{blue_cards}|{red_cards}|{neutral_card}"

            if state_hash in history:
                prev_move = history[state_hash]
                if move_count - prev_move > 20:
                    if log_session:
                        log_session.end_game(None, "cycle")
                    return DRAW, move_count

            history[state_hash] = move_count

        outcome = game.get_outcome()
        if outcome == ONGOING:
            outcome = DRAW

        # Log game end
        if log_session:
            winner = None
            if outcome == 1:  # BLUE_WINS
                winner = BLUE
            elif outcome == 2:  # RED_WINS
                winner = RED
            from src.game.serialization import determine_win_reason
            reason = determine_win_reason(game) or "max_moves"
            log_session.end_game(winner, reason)

        return outcome, move_count

    def _compile_participant_stats(
        self,
        matchup_results: List[MatchupResult],
        initial_elos: Dict[str, int]
    ) -> List[ParticipantStats]:
        """Compile aggregate statistics for each participant."""
        stats_map = {}

        for p in self.config.participants:
            stats_map[p] = ParticipantStats(
                participant=p,
                initial_elo=initial_elos[p],
                final_elo=self.elo.get_rating(p)
            )

        # Aggregate wins/losses/draws from matchups
        for result in matchup_results:
            a, b = result.participant_a, result.participant_b

            stats_map[a].total_wins += result.a_wins
            stats_map[a].total_losses += result.b_wins
            stats_map[a].total_draws += result.draws

            stats_map[b].total_wins += result.b_wins
            stats_map[b].total_losses += result.a_wins
            stats_map[b].total_draws += result.draws

        # Assign ranks by final Elo
        stats_list = list(stats_map.values())
        stats_list.sort(key=lambda s: s.final_elo, reverse=True)
        for i, stats in enumerate(stats_list, 1):
            stats.rank = i

        return stats_list
