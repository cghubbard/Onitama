"""
Debug version of main.py with added timeout and progress tracking.
"""
import argparse
import time
import random
import signal
import sys
from contextlib import contextmanager

# Import these directly to avoid circular imports
from src.utils.constants import BLUE, RED, PLAYER_NAMES, ONGOING, OUTCOME_NAMES
from src.game.game import Game  
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.utils.renderer import ConsoleRenderer, ASCIIRenderer


class TimeoutException(Exception):
    """Exception raised when a game times out."""
    pass


@contextmanager
def time_limit(seconds):
    """
    Context manager to limit execution time of a code block.
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def run_game(blue_agent_type, red_agent_type, renderer_type='ascii', delay=0.5, verbose=True, cards=None, game_timeout=30):
    """
    Run a game of Onitama between two agents with a timeout.
    
    Args:
        blue_agent_type: Type of agent for BLUE player ('random' or 'heuristic')
        red_agent_type: Type of agent for RED player ('random' or 'heuristic')
        renderer_type: Type of renderer to use ('console' or 'ascii')
        delay: Delay between moves in seconds (for visualization)
        verbose: Whether to print detailed information
        cards: Optional list of 5 card names to use
        game_timeout: Timeout in seconds for the entire game
        
    Returns:
        Game outcome or None if timed out
    """
    try:
        with time_limit(game_timeout):
            # Create the game
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
            max_moves = 200  # Safety limit to prevent infinite games
            
            if verbose:
                print("Starting game...")
                print(f"BLUE: {blue_agent_type}, RED: {red_agent_type}")
                print()
                renderer.render(game)
                print("\n")
            
            while game.get_outcome() == ONGOING and move_count < max_moves:
                current_player = game.get_current_player()
                agent = agents[current_player]
                
                # Get agent's move
                move = agent.select_move(game)
                if move is None:
                    print(f"No legal moves for {PLAYER_NAMES[current_player]}. Game ends in a draw.")
                    break
                
                from_pos, to_pos, card_name = move
                
                # Make the move
                success = game.make_move(from_pos, to_pos, card_name)
                move_count += 1
                
                if verbose:
                    print(f"Move {move_count}: {PLAYER_NAMES[current_player]} moves from {from_pos} to {to_pos} using {card_name}")
                    renderer.render(game)
                    print("\n")
                    time.sleep(delay)
            
            # Check if we hit the move limit
            if move_count >= max_moves:
                print(f"Game hit the maximum move count of {max_moves}. Forcing a draw.")
                return DRAW
            
            # Game over
            outcome = game.get_outcome()
            if verbose:
                print(f"Game over after {move_count} moves.")
                print(f"Outcome: {OUTCOME_NAMES[outcome]}")
            
            return outcome
    
    except TimeoutException as e:
        print(f"Game timed out: {e}")
        return None


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
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout in seconds for each game')
    parser.add_argument('--progress', action='store_true',
                        help='Show progress during multi-game runs')
    
    args = parser.parse_args()
    
    # Convert card names to proper format if provided
    cards = args.cards if args.cards is None else list(args.cards)
    
    # Run the specified number of games
    blue_wins = 0
    red_wins = 0
    draws = 0
    timeouts = 0
    
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
        
        outcome = run_game(
            blue_agent_type=args.blue,
            red_agent_type=args.red,
            renderer_type=args.renderer,
            delay=args.delay,
            verbose=not args.quiet,
            cards=cards,
            game_timeout=args.timeout
        )
        
        if outcome is None:
            timeouts += 1
            if args.progress:
                print(f"Game {i+1} timed out!")
        elif outcome == 1:  # BLUE_WINS
            blue_wins += 1
        elif outcome == 2:  # RED_WINS
            red_wins += 1
        else:  # DRAW
            draws += 1
    
    total_time = time.time() - start_time
    completed_games = blue_wins + red_wins + draws
    
    # Print statistics for multiple games
    if args.games > 1:
        print("\n===== Results =====")
        print(f"Games played: {args.games}")
        print(f"Games completed: {completed_games} ({completed_games/args.games:.1%})")
        if timeouts > 0:
            print(f"Games timed out: {timeouts} ({timeouts/args.games:.1%})")
        print(f"BLUE ({args.blue}) wins: {blue_wins} ({blue_wins/args.games:.1%})")
        print(f"RED ({args.red}) wins: {red_wins} ({red_wins/args.games:.1%})")
        print(f"Draws: {draws} ({draws/args.games:.1%})")
        print(f"Total time: {total_time:.2f}s, Average: {total_time/args.games:.4f}s per game")


if __name__ == "__main__":
    main()
