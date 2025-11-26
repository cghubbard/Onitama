# Onitama Game Implementation with AI Agents

This project implements the board game Onitama with various AI agents. It includes random and heuristic agents, along with a foundation for implementing a Proximal Policy Optimization (PPO) agent.

## Overview

Onitama is a two-player abstract strategy game where players control a Master and four Pawns on a 5x5 board. Players take turns moving a piece according to one of their two movement cards. After using a card, it is swapped with the neutral card. The objective is to either capture the opponent's Master or move your Master onto the opponent's shrine (center position of their back row).

## Project Structure

```
.
├── src/
│   ├── game/              # Core game implementation
│   │   ├── card.py        # Card class for move cards
│   │   └── game.py        # Game logic and rules
│   ├── agents/            # AI agents
│   │   ├── agent.py       # Abstract agent interface
│   │   ├── random_agent.py    # Random move agent
│   │   ├── heuristic_agent.py # Heuristic-based agent
│   │   └── ppo_agent.py       # PPO agent (placeholder implementation)
│   └── utils/             # Utility modules
│       ├── constants.py   # Game constants
│       ├── renderer.py    # Game rendering utilities
│       └── debug_utils.py # Debugging and visualization utilities
├── main.py                # Main script to run games
├── train_ppo.py           # Training script for PPO agent
├── debug_cards.py         # Debug script to visualize card movements
└── test_card_moves.py     # Test script to verify card moves for both players
```

## Features

- Complete Onitama game implementation with all rules
- Multiple AI agent implementations:
  - RandomAgent: Makes random legal moves
  - HeuristicAgent: Uses game heuristics to make strategic decisions
  - PPOAgent: Placeholder implementation for future reinforcement learning agent
- Console and ASCII-based game visualization
- Support for running multiple games for benchmarking
- Utilities for debugging and testing card movements

## Usage

### Run a single game:

```bash
python main.py
```

### Specify agent types:

```bash
python main.py --blue random --red heuristic
```

### Run multiple games to compare agent performance:

```bash
python main.py --blue random --red heuristic --games 100 --quiet
```

### Use specific cards:

```bash
python main.py --cards Tiger Dragon Frog Rabbit Crab
```

### For more options:

```bash
python main.py --help
```

### Visualize card movements:

To visualize a specific card's movement pattern:
```bash
python debug_cards.py Tiger
```

To visualize all cards:
```bash
python debug_cards.py
```

### Test card moves for both players:

```bash
python test_card_moves.py
```

### Train a PPO agent (placeholder functionality):

```bash
python train_ppo.py --episodes 10000 --opponent random --verbose
```

For more training options:
```bash
python train_ppo.py --help
```

## Example Game Output

```
Starting game...
BLUE: random, RED: heuristic

  0 1 2 3 4
 +-----------+
0| B b B b B |
1|           |
2|           |
3|           |
4| R r R r R |
 +-----------+

Current player: BLUE
Game state: ONGOING

BLUE cards:
  Tiger
  Dragon

RED cards:
  Frog
  Rabbit

Neutral card: Crab
```

## PPO Implementation Status

The project includes a placeholder implementation for a PPO (Proximal Policy Optimization) agent:

- The `PPOAgent` class in `src/agents/ppo_agent.py` provides the foundation for the reinforcement learning agent
- The `train_ppo.py` script contains the framework for training the PPO agent
- Current implementation uses random moves but includes architecture notes for future development

The planned PPO architecture includes:
1. State representation for neural network input
2. Policy and value networks with appropriate architectures
3. PPO-specific components like clipped objective function and entropy bonus
4. Training process with self-play or against other agents

## Future Improvements

- Complete the PPO agent implementation with neural networks
- Enhance the self-play training functionality
- Create a more sophisticated state representation for neural networks
- Implement board state visualization with a graphical interface
- Add Monte Carlo Tree Search (MCTS) implementation for comparison
- Implement curriculum learning for progressive agent training

## Requirements

- Python 3.8+
- numpy
- tqdm (for PPO training)
