#!/usr/bin/env python3
"""
Train linear value functions for Onitama.

This script trains a linear value function V(s) = θᵀφ̃(s) + b using
weighted logistic regression with L1/L2 regularization and cross-validation.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging.storage import GameStorage
from src.evaluation.data_loader import load_training_data
from src.evaluation.trainer import train_linear_value_function
from src.evaluation.model_store import ModelStore, LinearModel, NormalizationStats, TrainingInfo
from src.evaluation.weights import FEATURE_NAMES


def save_trained_model(
    name: str,
    result,  # TrainingResult
    dataset,  # TrainingDataset
    notes: str = ""
):
    """
    Save trained model to ModelStore.

    Args:
        name: Model name (e.g., 'trained_001')
        result: TrainingResult with weights and stats
        dataset: TrainingDataset used for training
        notes: Optional notes about training
    """
    # Create weight dictionary from FEATURE_NAMES
    weights_dict = {
        feat_name: float(weight)
        for feat_name, weight in zip(FEATURE_NAMES, result.theta)
    }

    # Create NormalizationStats
    norm_stats = NormalizationStats(
        means=result.means.tolist(),
        stds=result.stds.tolist(),
        epsilon=1e-8
    )

    # Create TrainingInfo
    training_info = TrainingInfo(
        data_source=f"{len(dataset.game_ids)} games",
        num_games=len(dataset.game_ids),
        gamma=dataset.gamma,
        lambda1=result.lambda1,
        lambda2=result.lambda2,
        val_loss=result.val_loss,
        train_loss=result.train_loss,
        notes=notes
    )

    # Create LinearModel
    model = LinearModel(
        name=name,
        weights=weights_dict,
        bias=float(result.bias),
        normalization=norm_stats,
        training=training_info,
        model_type="linear"
    )

    # Save to store
    store = ModelStore()
    store.save(model, notes=notes)

    return store._model_path(name)


def format_results_table(results):
    """Format grid search results as a table."""
    sorted_results = results.get_sorted_results(by='val_loss', ascending=True)

    lines = []
    lines.append("\nCV results (sorted by val_loss):")
    lines.append(f"{'λ₁':<10} {'λ₂':<10} {'Train Loss':<12} {'Val Loss':<12}")
    lines.append("-" * 50)

    for result in sorted_results[:10]:  # Show top 10
        lines.append(
            f"{result.lambda1:<10.4f} {result.lambda2:<10.4f} "
            f"{result.train_loss:<12.4f} {result.val_loss:<12.4f}"
        )

    if len(sorted_results) > 10:
        lines.append(f"... ({len(sorted_results) - 10} more models)")

    return "\n".join(lines)


def format_feature_weights(result):
    """Format feature weights as a table."""
    lines = []
    lines.append("\nFeature weights (best model):")
    lines.append(f"{'Feature':<35} {'Weight':<10}")
    lines.append("-" * 50)

    for feat_name, weight in zip(FEATURE_NAMES, result.theta):
        lines.append(f"{feat_name:<35} {weight:>10.2f}")

    lines.append(f"\nBias: {result.bias:.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Train linear value functions for Onitama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default training (4×4 grid, 5-fold CV)
  python scripts/train_linear.py --output trained_001

  # Custom regularization grid
  python scripts/train_linear.py \\
      --lambda1 0.0 0.01 0.1 1.0 10.0 \\
      --lambda2 0.0 0.01 0.1 1.0 10.0 \\
      --output trained_fine

  # Train on heuristic self-play only
  python scripts/train_linear.py \\
      --blue-agent heuristic \\
      --red-agent heuristic \\
      --limit 2000 \\
      --output trained_heuristic

  # Quick test with small grid
  python scripts/train_linear.py \\
      --limit 100 \\
      --lambda1 0.0 0.1 \\
      --lambda2 0.0 0.1 \\
      --cv-folds 3 \\
      --output test_run
'''
    )

    # Data options
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing game database (default: data)')
    parser.add_argument('--blue-agent', type=str, default=None,
                        help='Filter games by blue agent type (e.g., heuristic)')
    parser.add_argument('--red-agent', type=str, default=None,
                        help='Filter games by red agent type (e.g., heuristic)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Maximum number of games to use (default: all)')

    # Training options
    parser.add_argument('--gamma', type=float, default=0.97,
                        help='Discount factor for time-based weighting (default: 0.97)')
    parser.add_argument('--lambda1', type=float, nargs='+',
                        default=[0.0, 0.01, 0.1, 1.0],
                        help='L1 regularization values to try (default: 0.0 0.01 0.1 1.0)')
    parser.add_argument('--lambda2', type=float, nargs='+',
                        default=[0.0, 0.01, 0.1, 1.0],
                        help='L2 regularization values to try (default: 0.0 0.01 0.1 1.0)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Epsilon for feature standardization (default: 1e-8)')
    parser.add_argument('--exclude-draws', action='store_true',
                        help='Exclude drawn games from training')
    parser.add_argument('--agent-match-mode', type=str, default='contains',
                        choices=['exact', 'contains', 'prefix'],
                        help='How to match agent names: exact, contains (default), or prefix')
    parser.add_argument('--terminal-weight-multiplier', type=float, default=1.0,
                        help='Multiply terminal state weights by this factor (default: 1.0)')
    parser.add_argument('--terminal-only', action='store_true',
                        help='Only train on terminal states (default: False)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable feature normalization (use raw features)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    # Output options
    parser.add_argument('--output', type=str, required=True,
                        help='Name for the trained model (required)')
    parser.add_argument('--notes', type=str, default="",
                        help='Optional notes about this training run')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    verbose = not args.quiet

    # Print header
    if verbose:
        print("\n" + "=" * 60)
        print("Linear Value Function Training")
        print("=" * 60)

    # Load training data
    if verbose:
        print(f"\nLoading data from: {args.data_dir}/")
        filters = []
        if args.blue_agent:
            filters.append(f"blue_agent={args.blue_agent}")
        if args.red_agent:
            filters.append(f"red_agent={args.red_agent}")
        if filters:
            print(f"Filters: {', '.join(filters)}")
        if args.limit:
            print(f"Limit: {args.limit} games")

    try:
        storage = GameStorage(data_dir=args.data_dir)
        dataset = load_training_data(
            storage,
            blue_agent=args.blue_agent,
            red_agent=args.red_agent,
            limit=args.limit,
            gamma=args.gamma,
            exclude_draws=args.exclude_draws,
            verbose=verbose,
            agent_match_mode=args.agent_match_mode,
            terminal_weight_multiplier=args.terminal_weight_multiplier,
            terminal_only=args.terminal_only
        )
    except Exception as e:
        print(f"\nError loading data: {e}", file=sys.stderr)
        return 1

    if len(dataset) == 0:
        print("\nError: No training data found matching criteria", file=sys.stderr)
        return 1

    # Print dataset info
    if verbose:
        stats = dataset.get_statistics()
        print(f"\nDataset loaded:")
        print(f"  Games: {stats['n_games']}")
        print(f"  Examples: {stats['n_examples']}")
        print(f"  Wins: {stats['n_wins']} ({stats['win_rate']:.1%})")
        print(f"  Losses: {stats['n_losses']} ({1-stats['win_rate']:.1%})")
        print(f"  Gamma: {stats['gamma']}")
        print(f"  Total weight: {stats['total_weight']:.1f}")

    # Train model
    if verbose:
        print(f"\nGrid search: {len(args.lambda1)} × {len(args.lambda2)} = "
              f"{len(args.lambda1) * len(args.lambda2)} models")
        print(f"Cross-validation: {args.cv_folds} folds (game-level)")
        print("\nTraining...")

    try:
        results = train_linear_value_function(
            dataset,
            lambda1_values=args.lambda1,
            lambda2_values=args.lambda2,
            cv_folds=args.cv_folds,
            epsilon=args.epsilon,
            random_state=args.random_state,
            verbose=verbose,
            normalize=not args.no_normalize
        )
    except Exception as e:
        print(f"\nError during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Print results
    if verbose:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)

        print(f"\nBest model: λ₁={results.best_model.lambda1:.4f}, λ₂={results.best_model.lambda2:.4f}")
        print(f"  Train loss: {results.best_model.train_loss:.4f}")
        print(f"  Val loss:   {results.best_model.val_loss:.4f}")

        print(format_results_table(results))
        print(format_feature_weights(results.best_model))

    # Save model
    try:
        model_path = save_trained_model(
            name=args.output,
            result=results.best_model,
            dataset=dataset,
            notes=args.notes
        )

        if verbose:
            print(f"\nModel saved to: {model_path}")
            print("Registry updated.")
    except Exception as e:
        print(f"\nError saving model: {e}", file=sys.stderr)
        return 1

    if verbose:
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"\nTo test the model:")
        print(f"  python main.py --blue linear:{args.output} --red heuristic --games 100 --quiet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
