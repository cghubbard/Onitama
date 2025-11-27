"""
Default weights for the linear heuristic evaluation function.

V(s) = w^T * φ(s) where φ(s) is the feature vector and w is the weight vector.
"""

# Feature names in order (for logging/debugging)
FEATURE_NAMES = [
    'material_diff_students',
    'my_master_alive',
    'opp_master_captured',
    'master_safety_balance',
    'legal_moves_diff',
    'capture_moves_diff',
    'master_temple_distance_diff',
    'student_progress_diff',
    'central_control_diff',
    'card_mobility_diff',
    'master_escape_options',
]

# Default weights - hand-tuned starting point
DEFAULT_WEIGHTS = {
    'material_diff_students': 10.0,       # Each student worth ~10 points
    'my_master_alive': 1000.0,            # Critical - losing master = loss
    'opp_master_captured': 500.0,         # Opponent master dead = huge advantage
    'master_safety_balance': 15.0,        # Threats to masters are important
    'legal_moves_diff': 2.0,              # Mobility matters but less than material
    'capture_moves_diff': 5.0,            # Capture threats more valuable
    'master_temple_distance_diff': 8.0,   # Progress toward temple win
    'student_progress_diff': 3.0,         # Student advancement less critical
    'central_control_diff': 4.0,          # Board control
    'card_mobility_diff': 1.0,            # Card quality
    'master_escape_options': 3.0,         # Master flexibility
}

# As ordered list for dot product with feature vector
DEFAULT_WEIGHT_VECTOR = [DEFAULT_WEIGHTS[name] for name in FEATURE_NAMES]
