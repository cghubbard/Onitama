"""
Training script for PPO agent in Onitama.
"""
import argparse
import time
import os
import random
import numpy as np
import pickle
from tqdm import tqdm

from src.game.game import Game
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.ppo_agent import PPOAgent
from src.utils.constants import BLUE, RED, PLAYER_NAMES, ONGOING, OUTCOME_NAMES


def train_ppo_agent(
    num_episodes: int,
    opponent_type: str = 'random',
    save_path: str = 'models/ppo_agent.pkl',
    eval_every: int = 100,
    eval_episodes: int = 20,
    verbose: bool = True
):
    """
    Train a PPO agent on Onitama.
    
    Args:
        num_episodes: Number of episodes to train for
        opponent_type: Type of opponent to train against ('random', 'heuristic', 'self')
        save_path: Path to save the trained model
        eval_every: Evaluate the agent every this many episodes
        eval_episodes: Number of episodes to evaluate for
        verbose: Whether to print progress
    """
    # Create the agents
    ppo_agent = PPOAgent(BLUE)  # We'll train as BLUE
    
    # Create opponent
    if opponent_type == 'random':
        opponent = RandomAgent(RED)
    elif opponent_type == 'heuristic':
        opponent = HeuristicAgent(RED)
    elif opponent_type == 'self':
        opponent = PPOAgent(RED)  # Self-play
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    # Training loop
    best_win_rate = 0.0
    training_stats = {
        'episode_lengths': [],
        'episode_rewards': [],
        'win_rates': [],
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for episode in tqdm(range(1, num_episodes + 1)):
        # Reset the game
        game = Game()
        
        # Play until game over
        episode_length = 0
        episode_reward = 0
        
        while game.get_outcome() == ONGOING:
            current_player = game.get_current_player()
            
            if current_player == BLUE:
                # PPO agent's turn
                # In the future implementation, we would:
                # 1. Record the state
                # 2. Get action from policy
                # 3. Execute action
                # 4. Record reward
                # 5. Update PPO agent
                move = ppo_agent.select_move(game)
                if move is None:
                    break
                
                from_pos, to_pos, card_name = move
                game.make_move(from_pos, to_pos, card_name)
                
                # Simple reward structure (would be more sophisticated in actual implementation)
                if game.get_outcome() == BLUE:
                    episode_reward += 1.0  # Win
                
            else:
                # Opponent's turn
                move = opponent.select_move(game)
                if move is None:
                    break
                
                from_pos, to_pos, card_name = move
                game.make_move(from_pos, to_pos, card_name)
            
            episode_length += 1
        
        # Record statistics
        training_stats['episode_lengths'].append(episode_length)
        training_stats['episode_rewards'].append(episode_reward)
        
        # Evaluate agent
        if episode % eval_every == 0:
            win_rate = evaluate_agent(ppo_agent, opponent_type, eval_episodes, verbose)
            training_stats['win_rates'].append(win_rate)
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                ppo_agent.save_model(save_path)
                if verbose:
                    print(f"New best model saved with win rate: {win_rate:.1%}")
    
    # Save final model
    ppo_agent.save_model(save_path.replace('.pkl', '_final.pkl'))
    
    # Save training stats
    with open(save_path.replace('.pkl', '_stats.pkl'), 'wb') as f:
        pickle.dump(training_stats, f)
    
    return ppo_agent, training_stats


def evaluate_agent(agent, opponent_type: str, num_episodes: int, verbose: bool = False):
    """
    Evaluate an agent against a specified opponent.
    
    Args:
        agent: The agent to evaluate
        opponent_type: Type of opponent ('random', 'heuristic', 'self')
        num_episodes: Number of episodes to evaluate for
        verbose: Whether to print progress
        
    Returns:
        Win rate of the agent
    """
    # Create opponent
    if opponent_type == 'random':
        opponent = RandomAgent(RED)
    elif opponent_type == 'heuristic':
        opponent = HeuristicAgent(RED)
    elif opponent_type == 'self':
        opponent = PPOAgent(RED)  # Self-play
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    wins = 0
    
    for episode in range(num_episodes):
        game = Game()
        
        while game.get_outcome() == ONGOING:
            current_player = game.get_current_player()
            
            if current_player == BLUE:
                # Agent's turn
                move = agent.select_move(game)
                if move is None:
                    break
                from_pos, to_pos, card_name = move
                game.make_move(from_pos, to_pos, card_name)
            else:
                # Opponent's turn
                move = opponent.select_move(game)
                if move is None:
                    break
                from_pos, to_pos, card_name = move
                game.make_move(from_pos, to_pos, card_name)
        
        if game.get_outcome() == BLUE:
            wins += 1
    
    win_rate = wins / num_episodes
    if verbose:
        print(f"Evaluation: {wins}/{num_episodes} wins ({win_rate:.1%})")
    
    return win_rate


def main():
    """Parse command line arguments and run training."""
    parser = argparse.ArgumentParser(description='Train a PPO agent for Onitama.')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes to train for')
    parser.add_argument('--opponent', type=str, choices=['random', 'heuristic', 'self'], 
                        default='random', help='Type of opponent to train against')
    parser.add_argument('--save_path', type=str, default='models/ppo_agent.pkl',
                        help='Path to save the trained model')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Evaluate the agent every this many episodes')
    parser.add_argument('--eval_episodes', type=int, default=20,
                        help='Number of episodes to evaluate for')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress')
    
    args = parser.parse_args()
    
    train_ppo_agent(
        num_episodes=args.episodes,
        opponent_type=args.opponent,
        save_path=args.save_path,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
