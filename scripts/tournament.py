#!/usr/bin/env python3
"""
Round-robin tournament for Onitama agents with Elo rating calculation.

Usage:
    python scripts/tournament.py --participants random heuristic linear:baseline_v1 --games 500

Examples:
    # Quick test tournament
    python scripts/tournament.py --participants random heuristic --games 10

    # Full tournament with all trained models
    python scripts/tournament.py \\
        --participants random heuristic linear:baseline_v1 linear:trained_003_all_games \\
        --games 500 --update-models --progress

    # Minimal output
    python scripts/tournament.py --participants random heuristic linear --games 100 --quiet
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tournament.runner import TournamentRunner, TournamentConfig
from src.tournament.storage import TournamentStorage
from src.tournament.display import format_leaderboard, format_win_matrix
from src.evaluation.model_store import ModelStore
from main import parse_agent_spec, AGENT_TYPES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run a round-robin tournament between Onitama agents.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Agent specification formats:
  random              Random move selection
  heuristic           Simple heuristic agent
  linear              Linear agent with default weights
  linear:MODEL_NAME   Linear agent with specific model

Examples:
  python scripts/tournament.py --participants random heuristic linear --games 100
  python scripts/tournament.py --participants linear:baseline_v1 linear:trained_003_all_games --games 500
'''
    )

    parser.add_argument(
        '--participants', '-p',
        type=str, nargs='+', required=True,
        help='List of agent specs to compete (e.g., random heuristic linear:baseline_v1)'
    )
    parser.add_argument(
        '--games', '-g',
        type=int, default=500,
        help='Number of games per matchup (default: 500)'
    )
    parser.add_argument(
        '--k-factor',
        type=int, default=32,
        help='Elo K-factor for rating volatility (default: 32)'
    )
    parser.add_argument(
        '--initial-elo',
        type=int, default=1000,
        help='Default starting Elo for agents without existing rating (default: 1000)'
    )
    parser.add_argument(
        '--max-moves',
        type=int, default=200,
        help='Maximum moves per game before declaring a draw (default: 200)'
    )
    parser.add_argument(
        '--update-models',
        action='store_true',
        help='Update Elo ratings in ModelStore for model-based agents'
    )
    parser.add_argument(
        '--log',
        action='store_true',
        help='Enable full game logging (writes to data/games/)'
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show progress updates during tournament'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (only final results)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str, default=None,
        help='Tournament ID/name (auto-generated if not specified)'
    )
    parser.add_argument(
        '--data-dir',
        type=str, default='data',
        help='Directory for storing results (default: data)'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )

    return parser.parse_args()


def validate_participants(participants: list) -> bool:
    """Validate all participant specs before starting tournament."""
    errors = []

    for spec in participants:
        agent_type, model_name = parse_agent_spec(spec)

        if agent_type not in AGENT_TYPES:
            errors.append(f"Unknown agent type: {agent_type}")
            continue

        if agent_type == 'linear' and model_name:
            store = ModelStore()
            if not store.exists(model_name):
                available = [m.name for m in store.list_models()]
                errors.append(f"Model '{model_name}' not found. Available: {available}")

    if errors:
        for error in errors:
            print(f"Error: {error}")
        return False

    if len(participants) < 2:
        print("Error: Need at least 2 participants for a tournament")
        return False

    return True


def update_model_elos(result, participants: list):
    """Update Elo ratings in ModelStore for model-based agents."""
    store = ModelStore()

    for stats in result.participants:
        agent_type, model_name = parse_agent_spec(stats.participant)

        if model_name and store.exists(model_name):
            old_elo = stats.initial_elo
            new_elo = stats.final_elo
            store.update_elo(model_name, new_elo)
            print(f"Updated {model_name} Elo: {old_elo} -> {new_elo}")


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-models
    if args.list_models:
        store = ModelStore()
        models = store.list_models()
        if not models:
            print("No models found.")
        else:
            print("Available models:")
            print(f"{'Name':<24} {'Elo':<8} {'Notes'}")
            print("-" * 60)
            for m in models:
                elo_str = str(m.elo) if m.elo else "-"
                print(f"{m.name:<24} {elo_str:<8} {m.notes}")
        return 0

    # Validate participants
    if not validate_participants(args.participants):
        return 1

    # Create config
    config = TournamentConfig(
        participants=args.participants,
        games_per_matchup=args.games,
        k_factor=args.k_factor,
        initial_elo=args.initial_elo,
        max_moves=args.max_moves,
        log_games=args.log,
        data_dir=args.data_dir
    )

    # Run tournament
    runner = TournamentRunner(
        config=config,
        verbose=not args.quiet,
        show_progress=args.progress
    )

    result = runner.run(tournament_id=args.output)

    # Display final results
    print("\n" + format_leaderboard(result))
    print(format_win_matrix(result))

    # Update model Elo ratings if requested
    if args.update_models:
        print("\n--- Updating Model Elo Ratings ---")
        update_model_elos(result, args.participants)

    print(f"\nTournament ID: {result.tournament_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
