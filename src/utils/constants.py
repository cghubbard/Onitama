"""
Constants for the Onitama game.
"""

# Board dimensions
BOARD_SIZE = 5

# Players
BLUE = 0
RED = 1
PLAYER_NAMES = ["BLUE", "RED"]

# Piece types
PAWN = 0
MASTER = 1
PIECE_NAMES = ["PAWN", "MASTER"]

# Special positions
# Using (x, y) coordinates where x is column and y is row
BLUE_SHRINE = (2, 0)  # Top center
RED_SHRINE = (2, 4)   # Bottom center

# Game outcomes
ONGOING = 0
BLUE_WINS = 1
RED_WINS = 2
DRAW = 3  # If needed for stalemate scenarios
OUTCOME_NAMES = ["ONGOING", "BLUE_WINS", "RED_WINS", "DRAW"]

# Pre-defined move cards
# Format: [(dx, dy), ...] where:
# - dx: horizontal movement (positive = right, negative = left)
# - dy: vertical movement (negative = forward toward opponent, positive = backward)
#
# Movements are defined from RED's perspective (sitting at y=4, facing toward y=0).
# For BLUE, these movements are negated by the Card class.
#
# Example: Tiger [(0, -2), (0, 1)]
#   - (0, -2): Move 2 squares forward (toward opponent)
#   - (0, 1): Move 1 square backward (toward own side)
MOVE_CARDS = {
    "Tiger": [(0, -2), (0, 1)],                              # 2 forward, 1 back
    "Dragon": [(-2, -1), (2, -1), (-1, 1), (1, 1)],          # wide diagonal forward, close diagonal back
    "Frog": [(-2, 0), (-1, -1), (1, 1)],                     # 2 left, diag forward-left, diag back-right
    "Rabbit": [(2, 0), (1, -1), (-1, 1)],                    # 2 right, diag forward-right, diag back-left
    "Crab": [(0, -1), (-2, 0), (2, 0)],                      # 1 forward, 2 left, 2 right
    "Elephant": [(-1, -1), (1, -1), (-1, 0), (1, 0)],        # diag forward both sides, sides
    "Goose": [(-1, -1), (-1, 0), (1, 0), (1, 1)],            # forward-left, left, right, back-right
    "Rooster": [(1, -1), (1, 0), (-1, 0), (-1, 1)],          # forward-right, right, left, back-left
    "Monkey": [(-1, -1), (1, -1), (-1, 1), (1, 1)],          # all 4 diagonals
    "Mantis": [(-1, -1), (1, -1), (0, 1)],                   # 2 diag forward, 1 back
    "Horse": [(0, -1), (-1, 0), (0, 1)],                     # forward, left, back
    "Ox": [(0, -1), (1, 0), (0, 1)],                         # forward, right, back
    "Crane": [(0, -1), (-1, 1), (1, 1)],                     # 1 forward, 2 diag back
    "Boar": [(0, -1), (-1, 0), (1, 0)],                      # forward, left, right
    "Eel": [(-1, -1), (-1, 1), (1, 0)],                      # diag forward-left, diag back-left, right
    "Cobra": [(-1, 0), (1, -1), (1, 1)],                     # left, diag forward-right, diag back-right
}
