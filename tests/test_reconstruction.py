"""
Unit tests for game state reconstruction from snapshots.
"""
import pytest
from src.game.game import Game
from src.logging.trajectory import StateSnapshot, GameTrajectory, Transition, GameConfig, GameOutcome
from src.logging.reconstruction import reconstruct_game_from_snapshot, reconstruct_game_at_move
from src.game.serialization import serialize_board, serialize_game_state
from src.evaluation.features import FeatureExtractor
from src.utils.constants import BLUE, RED, PAWN, MASTER, ONGOING, BLUE_WINS


class TestGameReconstruction:
    """Tests for reconstructing Game objects from StateSnapshots."""

    @pytest.fixture
    def initial_game(self):
        """Create a game in starting position."""
        return Game(cards=['Tiger', 'Dragon', 'Frog', 'Rabbit', 'Crab'])

    @pytest.fixture
    def initial_snapshot(self, initial_game):
        """Create a snapshot of the initial game state."""
        return StateSnapshot(
            board=serialize_board(initial_game.board),
            current_player=initial_game.current_player,
            blue_cards=[card.name for card in initial_game.player_cards[BLUE]],
            red_cards=[card.name for card in initial_game.player_cards[RED]],
            neutral_card=initial_game.neutral_card.name,
            move_number=0,
            outcome=initial_game.outcome
        )

    @pytest.fixture
    def extractor(self):
        """Create a FeatureExtractor for testing."""
        return FeatureExtractor()

    # =========================================================================
    # Test 1: Basic reconstruction
    # =========================================================================
    def test_reconstruct_initial_state(self, initial_game, initial_snapshot):
        """Reconstructed game should match original initial state."""
        reconstructed = reconstruct_game_from_snapshot(initial_snapshot)

        # Check board state
        assert reconstructed.board == initial_game.board

        # Check current player
        assert reconstructed.current_player == initial_game.current_player

        # Check outcome
        assert reconstructed.outcome == initial_game.outcome

    def test_reconstructed_board_has_correct_pieces(self, initial_snapshot):
        """Reconstructed board should have all pieces in correct positions."""
        reconstructed = reconstruct_game_from_snapshot(initial_snapshot)

        # Count pieces
        blue_pieces = sum(1 for pos, (player, _) in reconstructed.board.items() if player == BLUE)
        red_pieces = sum(1 for pos, (player, _) in reconstructed.board.items() if player == RED)

        assert blue_pieces == 5  # 4 students + 1 master
        assert red_pieces == 5

        # Check master positions
        assert (2, 0) in reconstructed.board
        assert reconstructed.board[(2, 0)] == (BLUE, MASTER)
        assert (2, 4) in reconstructed.board
        assert reconstructed.board[(2, 4)] == (RED, MASTER)

    # =========================================================================
    # Test 2: Card state preservation
    # =========================================================================
    def test_cards_preserved(self, initial_game, initial_snapshot):
        """Cards should be correctly reconstructed."""
        reconstructed = reconstruct_game_from_snapshot(initial_snapshot)

        # Check player cards
        blue_card_names = [card.name for card in reconstructed.player_cards[BLUE]]
        red_card_names = [card.name for card in reconstructed.player_cards[RED]]
        original_blue = [card.name for card in initial_game.player_cards[BLUE]]
        original_red = [card.name for card in initial_game.player_cards[RED]]

        assert blue_card_names == original_blue
        assert red_card_names == original_red
        assert reconstructed.neutral_card.name == initial_game.neutral_card.name

    def test_card_movements_work(self, initial_snapshot):
        """Reconstructed cards should have working movement patterns."""
        reconstructed = reconstruct_game_from_snapshot(initial_snapshot)

        # Cards should have movement patterns
        for player in [BLUE, RED]:
            for card in reconstructed.player_cards[player]:
                assert len(card.movements) > 0
                # Each movement should be a (dx, dy) tuple
                for dx, dy in card.movements:
                    assert isinstance(dx, int)
                    assert isinstance(dy, int)

    # =========================================================================
    # Test 3: Feature extraction consistency
    # =========================================================================
    def test_features_match_original(self, initial_game, initial_snapshot, extractor):
        """Features extracted from reconstructed game should match original."""
        reconstructed = reconstruct_game_from_snapshot(initial_snapshot)

        original_features = extractor.extract(initial_game, BLUE)
        reconstructed_features = extractor.extract(reconstructed, BLUE)

        # Compare all 16 features
        assert original_features.material_diff_students == reconstructed_features.material_diff_students
        assert original_features.my_master_alive == reconstructed_features.my_master_alive
        assert original_features.opp_master_captured == reconstructed_features.opp_master_captured
        assert original_features.my_master_threats == reconstructed_features.my_master_threats
        assert original_features.opp_master_threats == reconstructed_features.opp_master_threats
        assert original_features.opp_shrine_threat == reconstructed_features.opp_shrine_threat
        assert original_features.my_legal_moves == reconstructed_features.my_legal_moves
        assert original_features.opp_legal_moves == reconstructed_features.opp_legal_moves
        assert original_features.my_capture_moves == reconstructed_features.my_capture_moves
        assert original_features.opp_capture_moves == reconstructed_features.opp_capture_moves
        assert original_features.my_master_temple_distance == reconstructed_features.my_master_temple_distance
        assert original_features.opp_master_temple_distance == reconstructed_features.opp_master_temple_distance
        assert original_features.my_student_progress == reconstructed_features.my_student_progress
        assert original_features.opp_student_progress == reconstructed_features.opp_student_progress
        assert original_features.central_control_diff == reconstructed_features.central_control_diff
        assert original_features.master_escape_options == reconstructed_features.master_escape_options

    def test_features_as_array_match(self, initial_game, initial_snapshot, extractor):
        """Feature arrays should match between original and reconstructed."""
        reconstructed = reconstruct_game_from_snapshot(initial_snapshot)

        original_array = extractor.extract_as_array(initial_game, BLUE)
        reconstructed_array = extractor.extract_as_array(reconstructed, BLUE)

        assert len(original_array) == len(reconstructed_array)
        assert original_array == reconstructed_array

    # =========================================================================
    # Test 4: Mid-game reconstruction
    # =========================================================================
    def test_reconstruct_after_moves(self, initial_game, extractor):
        """Reconstruction should work after game has progressed."""
        # Make a few moves
        moves_made = 0
        max_moves = 3
        while moves_made < max_moves and initial_game.outcome == ONGOING:
            legal_moves = initial_game.get_legal_moves(initial_game.current_player)
            if not legal_moves:
                break
            from_pos, to_pos, card_name = legal_moves[0]
            initial_game.make_move(from_pos, to_pos, card_name)
            moves_made += 1

        # Create snapshot of current state
        snapshot = StateSnapshot(
            board=serialize_board(initial_game.board),
            current_player=initial_game.current_player,
            blue_cards=[card.name for card in initial_game.player_cards[BLUE]],
            red_cards=[card.name for card in initial_game.player_cards[RED]],
            neutral_card=initial_game.neutral_card.name,
            move_number=moves_made,
            outcome=initial_game.outcome
        )

        # Reconstruct
        reconstructed = reconstruct_game_from_snapshot(snapshot)

        # Features should match
        original_features = extractor.extract_as_array(initial_game, BLUE)
        reconstructed_features = extractor.extract_as_array(reconstructed, BLUE)
        assert original_features == reconstructed_features

    # =========================================================================
    # Test 5: Edge cases
    # =========================================================================
    def test_reconstruct_with_captured_pieces(self):
        """Reconstruction should work when pieces have been captured."""
        # Create a snapshot with missing pieces (as if they were captured)
        snapshot = StateSnapshot(
            board={
                "2,0": [BLUE, MASTER],
                "1,0": [BLUE, PAWN],
                "3,0": [BLUE, PAWN],
                # Only BLUE master and 2 pawns remain (2 pawns captured)
                "2,4": [RED, MASTER],
                "0,4": [RED, PAWN],
                "1,4": [RED, PAWN],
                "3,4": [RED, PAWN],
                "4,4": [RED, PAWN],
                # All RED pieces remain
            },
            current_player=BLUE,
            blue_cards=["Tiger", "Dragon"],
            red_cards=["Frog", "Rabbit"],
            neutral_card="Crab",
            move_number=10,
            outcome=ONGOING
        )

        reconstructed = reconstruct_game_from_snapshot(snapshot)

        # Count pieces
        blue_pieces = sum(1 for pos, (player, _) in reconstructed.board.items() if player == BLUE)
        red_pieces = sum(1 for pos, (player, _) in reconstructed.board.items() if player == RED)

        assert blue_pieces == 3  # master + 2 pawns
        assert red_pieces == 5  # all pieces

    def test_reconstruct_with_master_on_shrine(self):
        """Reconstruction should work with master on opponent's shrine."""
        # Create a snapshot with BLUE master on RED shrine (winning position)
        snapshot = StateSnapshot(
            board={
                "2,4": [BLUE, MASTER],  # BLUE master on RED shrine
                "1,0": [BLUE, PAWN],
                "2,0": [RED, MASTER],
                "3,4": [RED, PAWN],
            },
            current_player=RED,
            blue_cards=["Tiger", "Dragon"],
            red_cards=["Frog", "Rabbit"],
            neutral_card="Crab",
            move_number=15,
            outcome=BLUE_WINS
        )

        reconstructed = reconstruct_game_from_snapshot(snapshot)

        # Check that BLUE master is on RED shrine
        assert (2, 4) in reconstructed.board
        assert reconstructed.board[(2, 4)] == (BLUE, MASTER)
        assert reconstructed.outcome == BLUE_WINS

    # =========================================================================
    # Test 6: Trajectory-based reconstruction
    # =========================================================================
    def test_reconstruct_at_specific_move(self):
        """Should be able to reconstruct game at specific move in trajectory."""
        # Create a simple trajectory with 3 transitions
        trajectory = GameTrajectory(
            game_id="test-123",
            timestamp="2025-01-01T00:00:00Z",
            config=GameConfig(
                cards_used=["Tiger", "Dragon", "Frog", "Rabbit", "Crab"],
                blue_agent="test",
                red_agent="test"
            )
        )

        # Add three snapshots
        for i in range(3):
            snapshot = StateSnapshot(
                board={
                    "2,0": [BLUE, MASTER],
                    f"{i},1": [BLUE, PAWN],  # Different pawn position each move
                    "2,4": [RED, MASTER],
                    "3,3": [RED, PAWN],
                },
                current_player=BLUE if i % 2 == 0 else RED,
                blue_cards=["Tiger", "Dragon"],
                red_cards=["Frog", "Rabbit"],
                neutral_card="Crab",
                move_number=i,
                outcome=ONGOING
            )

            transition = Transition(
                move_number=i,
                state=snapshot,
                legal_moves=[],
                action={"from": [0, 0], "to": [0, 0], "card": "Tiger"}
            )
            trajectory.add_transition(transition)

        # Reconstruct at move 1
        game_at_move_1 = reconstruct_game_at_move(trajectory, 1)
        assert (1, 1) in game_at_move_1.board
        assert game_at_move_1.board[(1, 1)] == (BLUE, PAWN)

        # Reconstruct at move 2
        game_at_move_2 = reconstruct_game_at_move(trajectory, 2)
        assert (2, 1) in game_at_move_2.board
        assert game_at_move_2.board[(2, 1)] == (BLUE, PAWN)

    def test_reconstruct_at_move_out_of_bounds(self):
        """Should raise IndexError for invalid move numbers."""
        trajectory = GameTrajectory(
            game_id="test-123",
            timestamp="2025-01-01T00:00:00Z"
        )

        # Add one transition
        snapshot = StateSnapshot(
            board={"2,0": [BLUE, MASTER]},
            current_player=BLUE,
            blue_cards=["Tiger", "Dragon"],
            red_cards=["Frog", "Rabbit"],
            neutral_card="Crab",
            move_number=0,
            outcome=ONGOING
        )
        transition = Transition(
            move_number=0,
            state=snapshot,
            legal_moves=[],
            action={"from": [0, 0], "to": [0, 0], "card": "Tiger"}
        )
        trajectory.add_transition(transition)

        # Negative index should raise
        with pytest.raises(IndexError):
            reconstruct_game_at_move(trajectory, -1)

        # Out of bounds should raise
        with pytest.raises(IndexError):
            reconstruct_game_at_move(trajectory, 10)

    # =========================================================================
    # Test 7: Legal moves work on reconstructed game
    # =========================================================================
    def test_legal_moves_on_reconstructed_game(self, initial_snapshot):
        """Should be able to get legal moves from reconstructed game."""
        reconstructed = reconstruct_game_from_snapshot(initial_snapshot)

        # Should be able to get legal moves
        legal_moves = reconstructed.get_legal_moves(BLUE)
        assert len(legal_moves) > 0

        # Each move should be a valid tuple
        for from_pos, to_pos, card_name in legal_moves:
            assert isinstance(from_pos, tuple)
            assert isinstance(to_pos, tuple)
            assert isinstance(card_name, str)

    def test_make_move_on_reconstructed_game(self, initial_snapshot):
        """Should be able to make moves on reconstructed game."""
        reconstructed = reconstruct_game_from_snapshot(initial_snapshot)

        # Get a legal move and execute it
        legal_moves = reconstructed.get_legal_moves(BLUE)
        assert len(legal_moves) > 0

        from_pos, to_pos, card_name = legal_moves[0]
        success = reconstructed.make_move(from_pos, to_pos, card_name)

        assert success
        assert reconstructed.current_player == RED  # Should switch to RED
