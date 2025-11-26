"""
Main script to run an Onitama game between two agents.
"""
import argparse
import time
import random
import sys

# Import these directly to avoid circular imports
from src.utils.constants import BLUE, RED, PLAYER_NAMES, ONGOING, OUTCOME_NAMES, DRAW
from src.game.game import Game
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.utils.renderer import ConsoleRenderer, ASCIIRenderer
from src.logging.game_logger import GameLogger, GameLogSession, create_logger_from_args
from src.game.serialization import determine_win_reason


def run_game(blue_agent_type, red_agent_type, renderer_type='ascii', delay=0.5, verbose=True, cards=None, max_moves=200, log_session=None, game=None):
    """
    Run a game of Onitama between two agents.

    Args:
        blue_agent_type: Type of agent for BLUE player ('random' or 'heuristic')
        red_agent_type: Type of agent for RED player ('random' or 'heuristic')
        renderer_type: Type of renderer to use ('console' or 'ascii')
        delay: Delay between moves in seconds (for visualization)
        verbose: Whether to print detailed information
        cards: Optional list of 5 card names to use
        max_moves: Maximum number of moves before forcing a draw
        log_session: Optional GameLogSession for logging the game
        game: Optional pre-created Game instance

    Returns:
        Game outcome
    """
    # Create the game if not provided
    if game is None:
        game = Game(cards=cards)
    
    # Create the agents
    agents = {}
    if blue_agent_type.lower() == 'random':
        agents[BLUE] = RandomAgent(BLUE)
    else:  # heuristic
        agents[BLUE] = HeuristicAgent(BLUE)
    
    if red_agent_type.lower() == 'random':
        agents[RED] = RandomAgent(RED)
    else:  # heuristic
        agents[RED] = HeuristicAgent(RED)
    
    # Create the renderer
    if renderer_type.lower() == 'console':
        renderer = ConsoleRenderer
    else:  # ascii
        renderer = ASCIIRenderer
    
    # Game loop
    move_count = 0
    
    # Track game state to detect cycles
    history = {}  # Maps board state hash to move number
    
    if verbose:
        print("Starting game...")
        print(f"BLUE: {blue_agent_type}, RED: {red_agent_type}")
        print()
        renderer.render(game)
        print("\n")
    
    while game.get_outcome() == ONGOING:
        # Check if we've exceeded the maximum move count
        if move_count >= max_moves:
            if verbose:
                print(f"Game has reached the maximum move count of {max_moves}. Ending as a draw.")
            if log_session:
                log_session.end_game(None, "max_moves")
            return DRAW

        current_player = game.get_current_player()
        agent = agents[current_player]

        # Get agent's move
        move = agent.select_move(game)
        if move is None:
            if verbose:
                print(f"No legal moves for {PLAYER_NAMES[current_player]}. Game ends in a draw.")
            if log_session:
                log_session.end_game(None, "draw")
            return DRAW

        from_pos, to_pos, card_name = move

        # Log pre-move state if logging
        pre_state = None
        if log_session:
            pre_state = log_session.log_pre_move_state(game)

        # Make the move
        success = game.make_move(from_pos, to_pos, card_name)
        move_count += 1

        # Log the move after it's made
        if log_session and pre_state:
            log_session.log_move_with_pre_state(pre_state, move)
        
        # Generate a simple hash of the board state to detect cycles
        # Include current player and card distribution
        board_hash = str(sorted(game.get_board_state().items()))
        player_hash = str(game.get_current_player())
        blue_cards = str(sorted(card.name for card in game.get_player_cards(BLUE)))
        red_cards = str(sorted(card.name for card in game.get_player_cards(RED)))
        neutral_card = game.get_neutral_card().name
        state_hash = f"{board_hash}|{player_hash}|{blue_cards}|{red_cards}|{neutral_card}"
        
        # Check if we've seen this exact state before (potential cycle)
        if state_hash in history:
            prev_move = history[state_hash]
            # If we've seen this state and it was more than 20 moves ago, declare a draw
            # This prevents small oscillations from immediately triggering the cycle detection
            if move_count - prev_move > 20:
                if verbose:
                    print(f"Cycle detected between moves {prev_move} and {move_count}. Ending as a draw.")
                if log_session:
                    log_session.end_game(None, "draw")
                return DRAW
        
        # Store the current state
        history[state_hash] = move_count
        
        if verbose:
            print(f"Move {move_count}: {PLAYER_NAMES[current_player]} moves from {from_pos} to {to_pos} using {card_name}")
            renderer.render(game)
            print("\n")
            time.sleep(delay)
    
    # Game over
    outcome = game.get_outcome()
    if verbose:
        print(f"Game over after {move_count} moves.")
        print(f"Outcome: {OUTCOME_NAMES[outcome]}")

    # Log game end
    if log_session:
        winner = None
        if outcome == 1:  # BLUE_WINS
            winner = BLUE
        elif outcome == 2:  # RED_WINS
            winner = RED
        reason = determine_win_reason(game) or "unknown"
        log_session.end_game(winner, reason)

    return outcome


def main():
    """Main function to parse arguments and run the game."""
    parser = argparse.ArgumentParser(description='Run an Onitama game between two agents.')
    parser.add_argument('--blue', type=str, choices=['random', 'heuristic'], default='random',
                        help='Type of agent for BLUE player')
    parser.add_argument('--red', type=str, choices=['random', 'heuristic'], default='heuristic',
                        help='Type of agent for RED player')
    parser.add_argument('--renderer', type=str, choices=['console', 'ascii'], default='ascii',
                        help='Type of renderer to use')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Delay between moves in seconds')
    parser.add_argument('--quiet', action='store_true',
                        help='Run in quiet mode (no output)')
    parser.add_argument('--games', type=int, default=1,
                        help='Number of games to run')
    parser.add_argument('--cards', type=str, nargs=5,
                        help='Five specific cards to use (e.g., Tiger Dragon Frog Rabbit Crab)')
    parser.add_argument('--max-moves', type=int, default=200,
                        help='Maximum number of moves before forcing a draw')
    parser.add_argument('--progress', action='store_true',
                        help='Show progress during multi-game runs')
    parser.add_argument('--log', type=str, choices=['none', 'all', 'sample'], default='none',
                        help='Game logging mode: none, all, or sample')
    parser.add_argument('--sample-rate', type=float, default=0.1,
                        help='Sampling rate for --log sample mode (default: 0.1 = 10%%)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory for storing game logs')

    args = parser.parse_args()

    # Create game logger
    logger = create_logger_from_args(args.log, args.sample_rate, args.data_dir)
    
    # Convert card names to proper format if provided
    cards = args.cards if args.cards is None else list(args.cards)

    # Run the specified number of games
    blue_wins = 0
    red_wins = 0
    draws = 0
    games_logged = 0

    start_time = time.time()

    for i in range(args.games):
        if args.progress and i > 0 and i % 100 == 0:
            elapsed = time.time() - start_time
            games_per_second = i / elapsed if elapsed > 0 else 0
            eta = (args.games - i) / games_per_second if games_per_second > 0 else "unknown"
            print(f"Progress: {i}/{args.games} games ({i/args.games:.1%}), "
                  f"Rate: {games_per_second:.2f} games/s, ETA: {eta:.1f}s")

        if args.games > 1 and not args.quiet and not args.progress:
            print(f"\n===== Game {i+1} =====\n")

        # Create a game to get the log session started
        game = Game(cards=cards)
        log_session = logger.start_game(game, args.blue, args.red)
        if log_session.active:
            games_logged += 1

        outcome = run_game(
            blue_agent_type=args.blue,
            red_agent_type=args.red,
            renderer_type=args.renderer,
            delay=args.delay,
            verbose=not args.quiet,
            cards=cards,
            max_moves=args.max_moves,
            log_session=log_session,
            game=game
        )
        
        if outcome == 1:  # BLUE_WINS
            blue_wins += 1
        elif outcome == 2:  # RED_WINS
            red_wins += 1
        else:  # DRAW
            draws += 1
    
    total_time = time.time() - start_time
    
    # Print statistics for multiple games
    if args.games > 1:
        print("\n===== Results =====")
        print(f"Games played: {args.games}")
        print(f"BLUE ({args.blue}) wins: {blue_wins} ({blue_wins/args.games:.1%})")
        print(f"RED ({args.red}) wins: {red_wins} ({red_wins/args.games:.1%})")
        print(f"Draws: {draws} ({draws/args.games:.1%})")
        print(f"Total time: {total_time:.2f}s, Average: {total_time/args.games:.4f}s per game")
        if args.log != 'none':
            print(f"Games logged: {games_logged}")


if __name__ == "__main__":
    main()
