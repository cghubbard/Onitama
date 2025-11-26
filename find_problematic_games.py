"""
Script to identify problematic game scenarios by tracking execution time and outcomes.
"""
import argparse
import time
import random
from collections import defaultdict

from src.utils.constants import BLUE, RED, PLAYER_NAMES, ONGOING, OUTCOME_NAMES, MOVE_CARDS
from src.game.game import Game  
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent


def run_game_with_stats(blue_agent_type, red_agent_type, cards=None, max_moves=500):
    """
    Run a game and collect statistics about its execution.
    
    Args:
        blue_agent_type: Type of agent for BLUE player ('random' or 'heuristic')
        red_agent_type: Type of agent for RED player ('random' or 'heuristic')
        cards: Optional list of 5 card names to use
        max_moves: Maximum number of moves before forcing a draw
        
    Returns:
        Dictionary with execution statistics
    """
    start_time = time.time()
    
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
    
    # Game loop
    move_count = 0
    moves_by_player = {BLUE: 0, RED: 0}
    blue_cards = [card.name for card in game.get_player_cards(BLUE)]
    red_cards = [card.name for card in game.get_player_cards(RED)]
    neutral_card = game.get_neutral_card().name
    
    while game.get_outcome() == ONGOING and move_count < max_moves:
        current_player = game.get_current_player()
        agent = agents[current_player]
        
        # Record move start time to track agent decision making
        move_start_time = time.time()
        
        # Get agent's move
        move = agent.select_move(game)
        
        move_time = time.time() - move_start_time
        
        if move is None:
            break
        
        from_pos, to_pos, card_name = move
        
        # Make the move
        success = game.make_move(from_pos, to_pos, card_name)
        move_count += 1
        moves_by_player[current_player] += 1
    
    # Game over
    outcome = game.get_outcome()
    if move_count >= max_moves:
        outcome = 3  # Force DRAW
    
    total_time = time.time() - start_time
    
    return {
        "outcome": outcome,
        "total_time": total_time,
        "move_count": move_count,
        "moves_by_player": moves_by_player,
        "cards": {
            "blue": blue_cards,
            "red": red_cards,
            "neutral": neutral_card
        },
        "avg_time_per_move": total_time / max(1, move_count)
    }


def main():
    """Main function to analyze game execution patterns."""
    parser = argparse.ArgumentParser(description='Find problematic game scenarios in Onitama.')
    parser.add_argument('--blue', type=str, choices=['random', 'heuristic'], default='random',
                        help='Type of agent for BLUE player')
    parser.add_argument('--red', type=str, choices=['random', 'heuristic'], default='heuristic',
                        help='Type of agent for RED player')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games to analyze')
    parser.add_argument('--max-moves', type=int, default=500,
                        help='Maximum moves per game before forcing a draw')
    parser.add_argument('--time-threshold', type=float, default=0.5,
                        help='Time threshold in seconds to identify slow games')
    parser.add_argument('--card-analysis', action='store_true',
                        help='Analyze which cards appear in slow games')
    
    args = parser.parse_args()
    
    # Statistics
    game_times = []
    long_games = []
    move_counts = []
    card_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "long_games": 0})
    
    print(f"Analyzing {args.games} games with {args.blue} (BLUE) vs {args.red} (RED)...")
    start_time = time.time()
    
    for i in range(args.games):
        if i > 0 and i % 10 == 0:
            print(f"Progress: {i}/{args.games} games ({i/args.games:.1%})")
        
        stats = run_game_with_stats(
            blue_agent_type=args.blue,
            red_agent_type=args.red,
            max_moves=args.max_moves
        )
        
        game_times.append(stats["total_time"])
        move_counts.append(stats["move_count"])
        
        # Track long games
        if stats["total_time"] > args.time_threshold:
            long_games.append({
                "game_number": i + 1,
                "time": stats["total_time"],
                "moves": stats["move_count"],
                "cards": stats["cards"]
            })
        
        # Card analysis
        if args.card_analysis:
            all_cards = (
                stats["cards"]["blue"] + 
                stats["cards"]["red"] + 
                [stats["cards"]["neutral"]]
            )
            
            for card in all_cards:
                card_stats[card]["count"] += 1
                card_stats[card]["total_time"] += stats["total_time"]
                if stats["total_time"] > args.time_threshold:
                    card_stats[card]["long_games"] += 1
    
    total_analysis_time = time.time() - start_time
    
    # Print analysis results
    print("\n===== Game Time Analysis =====")
    print(f"Total games analyzed: {args.games}")
    print(f"Total analysis time: {total_analysis_time:.2f}s")
    print(f"Average time per game: {sum(game_times)/len(game_times):.4f}s")
    print(f"Fastest game: {min(game_times):.4f}s")
    print(f"Slowest game: {max(game_times):.4f}s")
    print(f"Long games (>{args.time_threshold}s): {len(long_games)} ({len(long_games)/args.games:.1%})")
    
    print("\n===== Move Count Analysis =====")
    print(f"Average moves per game: {sum(move_counts)/len(move_counts):.1f}")
    print(f"Min moves: {min(move_counts)}")
    print(f"Max moves: {max(move_counts)}")
    print(f"Games hitting max moves: {sum(1 for x in move_counts if x >= args.max_moves)}")
    
    if args.card_analysis:
        print("\n===== Card Analysis =====")
        print("Cards appearing in long games (sorted by frequency):")
        
        # Sort cards by frequency in long games
        sorted_cards = sorted(
            card_stats.items(), 
            key=lambda x: (x[1]["long_games"], x[1]["total_time"]), 
            reverse=True
        )
        
        for card, stats in sorted_cards:
            if stats["long_games"] > 0:
                avg_time = stats["total_time"] / stats["count"]
                long_game_pct = stats["long_games"] / stats["count"] * 100
                print(f"{card}: {stats['long_games']} long games ({long_game_pct:.1f}%), "
                      f"avg time: {avg_time:.4f}s")
    
    if long_games:
        print("\n===== Details of Top 5 Longest Games =====")
        for game in sorted(long_games, key=lambda x: x["time"], reverse=True)[:5]:
            print(f"Game {game['game_number']}: {game['time']:.4f}s, {game['moves']} moves")
            print(f"  BLUE cards: {', '.join(game['cards']['blue'])}")
            print(f"  RED cards: {', '.join(game['cards']['red'])}")
            print(f"  Neutral card: {game['cards']['neutral']}")
            print()


if __name__ == "__main__":
    main()
