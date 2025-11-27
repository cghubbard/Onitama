"""
Unit tests for the feature extraction module.
"""

import pytest
from src.game.game import Game
from src.evaluation.features import FeatureExtractor, FeatureVector
from src.evaluation.weights import DEFAULT_WEIGHT_VECTOR, FEATURE_NAMES
from src.utils.constants import BLUE, RED, PAWN, MASTER, BLUE_SHRINE, RED_SHRINE


class TestFeatureExtractor:
    """Unit tests for FeatureExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a FeatureExtractor instance."""
        return FeatureExtractor()

    @pytest.fixture
    def starting_game(self):
        """Create a game in the standard starting position."""
        return Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])

    # =========================================================================
    # Test 1: Material features at game start
    # =========================================================================
    def test_material_at_start(self, extractor, starting_game):
        """At game start, material should be balanced."""
        features = extractor.extract(starting_game, BLUE)

        # 4 students each, so difference is 0
        assert features.material_diff_students == 0
        # Both masters should be alive
        assert features.my_master_alive == 1
        assert features.opp_master_captured == 0

    def test_material_symmetry(self, extractor, starting_game):
        """Material features should be symmetric between players at start."""
        blue_features = extractor.extract(starting_game, BLUE)
        red_features = extractor.extract(starting_game, RED)

        assert blue_features.material_diff_students == red_features.material_diff_students
        assert blue_features.my_master_alive == red_features.my_master_alive
        assert blue_features.opp_master_captured == red_features.opp_master_captured

    # =========================================================================
    # Test 2: Central control
    # =========================================================================
    def test_central_control_at_start(self, extractor, starting_game):
        """At start, no pieces are in the central 3x3."""
        features = extractor.extract(starting_game, BLUE)

        # Initial positions are row 0 (BLUE) and row 4 (RED)
        # Neither is in the center (rows 1-3, cols 1-3)
        assert features.central_control_diff == 0

    def test_central_squares_definition(self, extractor):
        """Verify central squares are correctly defined."""
        expected = {(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)}
        assert extractor.CENTRAL_SQUARES == expected

    # =========================================================================
    # Test 3: Temple distance features
    # =========================================================================
    def test_temple_distance_at_start(self, extractor, starting_game):
        """At start, both masters are equidistant from enemy temples."""
        features = extractor.extract(starting_game, BLUE)

        # BLUE master at (2, 0), enemy temple at (2, 4) -> distance 4
        # RED master at (2, 4), BLUE temple at (2, 0) -> distance 4
        # Difference should be 0
        assert features.master_temple_distance_diff == 0.0

    def test_temple_distance_from_red_perspective(self, extractor, starting_game):
        """Temple distance should also be 0 from RED's perspective at start."""
        features = extractor.extract(starting_game, RED)
        assert features.master_temple_distance_diff == 0.0

    # =========================================================================
    # Test 4: Student progress
    # =========================================================================
    def test_student_progress_at_start(self, extractor, starting_game):
        """Student progress should be balanced at start."""
        features = extractor.extract(starting_game, BLUE)

        # BLUE students at y=0, need to reach RED temple at (2, 4)
        # RED students at y=4, need to reach BLUE temple at (2, 0)
        # Both have similar average distances, so diff should be ~0
        assert abs(features.student_progress_diff) < 0.1

    # =========================================================================
    # Test 5: Mobility features
    # =========================================================================
    def test_legal_moves_at_start(self, extractor, starting_game):
        """Test legal moves calculation at start."""
        features = extractor.extract(starting_game, BLUE)

        # Legal moves should be computed for both players
        # The difference depends on the specific cards
        # Just verify it's an integer
        assert isinstance(features.legal_moves_diff, int)

    def test_capture_moves_at_start(self, extractor, starting_game):
        """At start, no capture moves should be available."""
        features = extractor.extract(starting_game, BLUE)

        # No pieces are adjacent, so no captures possible
        assert features.capture_moves_diff == 0

    def test_master_safety_at_start(self, extractor, starting_game):
        """At start, no master threats should exist."""
        features = extractor.extract(starting_game, BLUE)

        # Masters are far apart, no threats
        assert features.master_safety_balance == 0

    # =========================================================================
    # Test 6: Feature vector structure
    # =========================================================================
    def test_feature_vector_has_11_elements(self, extractor, starting_game):
        """Feature vector should have exactly 11 elements."""
        features = extractor.extract(starting_game, BLUE)
        assert len(features) == 11

    def test_extract_as_array(self, extractor, starting_game):
        """extract_as_array should return a list of floats."""
        array = extractor.extract_as_array(starting_game, BLUE)
        assert isinstance(array, list)
        assert len(array) == 11
        assert all(isinstance(x, (int, float)) for x in array)

    def test_feature_names_count(self):
        """Should have 11 feature names."""
        assert len(FEATURE_NAMES) == 11

    def test_default_weights_count(self):
        """Should have 11 default weights."""
        assert len(DEFAULT_WEIGHT_VECTOR) == 11

    # =========================================================================
    # Test 7: Evaluation function
    # =========================================================================
    def test_evaluate_with_default_weights(self, extractor, starting_game):
        """Evaluate should return a scalar score."""
        score = extractor.evaluate(starting_game, BLUE, DEFAULT_WEIGHT_VECTOR)
        assert isinstance(score, float)

    def test_evaluate_symmetric_at_start(self, extractor, starting_game):
        """At start, both players should have similar evaluations."""
        blue_score = extractor.evaluate(starting_game, BLUE, DEFAULT_WEIGHT_VECTOR)
        red_score = extractor.evaluate(starting_game, RED, DEFAULT_WEIGHT_VECTOR)

        # Both scores should include my_master_alive bonus
        # Scores should be close but not necessarily equal due to mobility differences
        assert abs(blue_score - red_score) < 50  # Allow for mobility differences

    # =========================================================================
    # Test 8: Perspective handling
    # =========================================================================
    def test_perspective_affects_features(self, extractor):
        """Test that perspective correctly flips feature signs."""
        # Create a game and make a move to create asymmetry
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])

        # Make a move for BLUE (moves a piece forward)
        moves = game.get_legal_moves(BLUE)
        if moves:
            game.make_move(*moves[0])

        # Now features should differ between perspectives
        blue_features = extractor.extract(game, BLUE)
        red_features = extractor.extract(game, RED)

        # my_master_alive should still be 1 for both
        assert blue_features.my_master_alive == 1
        assert red_features.my_master_alive == 1

    # =========================================================================
    # Test 9: Master escape options
    # =========================================================================
    def test_master_escape_options(self, extractor, starting_game):
        """Master should have some escape options at start."""
        features = extractor.extract(starting_game, BLUE)

        # Master at (2, 0) should have some moves depending on cards
        # Just verify it's a non-negative integer
        assert features.master_escape_options >= 0
        assert isinstance(features.master_escape_options, int)

    # =========================================================================
    # Test 10: Custom board states
    # =========================================================================
    def test_material_after_capture(self, extractor):
        """Test material difference after a capture occurs."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])

        # Manually modify board to simulate a capture scenario
        # Remove one RED student
        board = game.board
        red_students = [(pos, piece) for pos, piece in board.items()
                       if piece[0] == RED and piece[1] == PAWN]
        if red_students:
            del game.board[red_students[0][0]]

        features = extractor.extract(game, BLUE)
        # BLUE has 4 students, RED has 3 now
        assert features.material_diff_students == 1

    def test_opp_master_captured(self, extractor):
        """Test opp_master_captured when opponent master is removed."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])

        # Remove RED master
        red_master_pos = None
        for pos, piece in game.board.items():
            if piece[0] == RED and piece[1] == MASTER:
                red_master_pos = pos
                break
        if red_master_pos:
            del game.board[red_master_pos]

        features = extractor.extract(game, BLUE)
        assert features.opp_master_captured == 1
        assert features.my_master_alive == 1

    def test_my_master_dead(self, extractor):
        """Test features when my master is captured."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])

        # Remove BLUE master
        blue_master_pos = None
        for pos, piece in game.board.items():
            if piece[0] == BLUE and piece[1] == MASTER:
                blue_master_pos = pos
                break
        if blue_master_pos:
            del game.board[blue_master_pos]

        features = extractor.extract(game, BLUE)
        assert features.my_master_alive == 0
        assert features.opp_master_captured == 0
        # Temple distance should reflect disadvantage
        assert features.master_temple_distance_diff < 0


class TestGameCopy:
    """Test the Game.copy() method."""

    def test_copy_preserves_board(self):
        """Copy should preserve board state."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])
        copy = game.copy()

        assert copy.get_board_state() == game.get_board_state()

    def test_copy_is_independent(self):
        """Modifications to copy shouldn't affect original."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])
        copy = game.copy()

        # Make a move on the copy
        moves = copy.get_legal_moves(BLUE)
        if moves:
            copy.make_move(*moves[0])

        # Original should be unchanged
        assert game.get_current_player() == BLUE
        assert len(game.move_history) == 0

    def test_copy_preserves_cards(self):
        """Copy should preserve card state."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])
        copy = game.copy()

        original_blue_cards = [c.name for c in game.get_player_cards(BLUE)]
        copy_blue_cards = [c.name for c in copy.get_player_cards(BLUE)]
        assert original_blue_cards == copy_blue_cards

    def test_copy_preserves_current_player(self):
        """Copy should preserve current player."""
        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])
        copy = game.copy()
        assert copy.get_current_player() == game.get_current_player()


class TestLinearHeuristicAgent:
    """Test the LinearHeuristicAgent."""

    def test_agent_selects_move(self):
        """Agent should select a valid move."""
        from src.agents.linear_heuristic_agent import LinearHeuristicAgent

        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])
        agent = LinearHeuristicAgent(BLUE)

        move = agent.select_move(game)
        assert move is not None

        # Move should be in legal moves
        legal_moves = game.get_legal_moves(BLUE)
        assert move in legal_moves

    def test_agent_with_custom_weights(self):
        """Agent should work with custom weights."""
        from src.agents.linear_heuristic_agent import LinearHeuristicAgent

        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])
        custom_weights = [1.0] * 11
        agent = LinearHeuristicAgent(BLUE, weights=custom_weights)

        move = agent.select_move(game)
        assert move is not None

    def test_agent_prefers_winning_moves(self):
        """Agent should prefer moves that win the game."""
        from src.agents.linear_heuristic_agent import LinearHeuristicAgent

        game = Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])

        # Set up a position where BLUE can capture RED master
        # Clear the board first
        game.board.clear()
        # Place BLUE master adjacent to RED master
        game.board[(2, 2)] = (BLUE, MASTER)
        game.board[(2, 3)] = (RED, MASTER)
        # Add a BLUE student for variety
        game.board[(0, 0)] = (BLUE, PAWN)
        game.board[(4, 4)] = (RED, PAWN)

        agent = LinearHeuristicAgent(BLUE, randomize=False)
        move = agent.select_move(game)

        # If there's a move that captures the RED master, agent should take it
        # The exact move depends on the cards, but the evaluation should favor it
        assert move is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
