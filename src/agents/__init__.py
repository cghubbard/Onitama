"""
Agents module for Onitama implementation.
"""
from src.agents.agent import Agent
from src.agents.random_agent import RandomAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.ppo_agent import PPOAgent

__all__ = ['Agent', 'RandomAgent', 'HeuristicAgent', 'PPOAgent']
