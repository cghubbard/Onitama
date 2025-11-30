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
    'my_legal_moves',
    'opp_legal_moves',
    'my_capture_moves',
    'opp_capture_moves',
    'my_master_temple_distance',
    'opp_master_temple_distance',
    'my_student_progress',
    'opp_student_progress',
    'central_control_diff',
    'master_escape_options',
]

# Default weights - hand-tuned starting point
# Note: For split features, negative weights mean "lower is better"
DEFAULT_WEIGHTS = {
    'material_diff_students': 10.0,        # Each student worth ~10 points
    'my_master_alive': 1000.0,             # Critical - losing master = loss
    'opp_master_captured': 500.0,          # Opponent master dead = huge advantage
    'master_safety_balance': 15.0,         # Threats to masters are important
    'my_legal_moves': 2.0,                 # More options = better
    'opp_legal_moves': -2.0,               # More opponent options = worse for me
    'my_capture_moves': 5.0,               # Capture opportunities are valuable
    'opp_capture_moves': -5.0,             # Opponent capture threats = danger
    'my_master_temple_distance': -8.0,     # Lower distance = closer to win
    'opp_master_temple_distance': 8.0,     # Higher opponent distance = safer
    'my_student_progress': -3.0,           # Lower distance = more advanced
    'opp_student_progress': 3.0,           # Higher opponent distance = safer
    'central_control_diff': 4.0,           # Board control
    'master_escape_options': 3.0,          # Master flexibility
}

# As ordered list for dot product with feature vector
DEFAULT_WEIGHT_VECTOR = [DEFAULT_WEIGHTS[name] for name in FEATURE_NAMES]
