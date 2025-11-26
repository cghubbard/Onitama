"""
Game logging module for Onitama.

Provides RL-ready game trajectory logging with configurable storage.
"""
from src.logging.trajectory import StateSnapshot, Transition, GameTrajectory
from src.logging.game_logger import GameLogger
from src.logging.storage import GameStorage

__all__ = [
    'StateSnapshot',
    'Transition',
    'GameTrajectory',
    'GameLogger',
    'GameStorage'
]
