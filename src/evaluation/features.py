"""
Feature extraction for Onitama board positions.

Extracts features for the linear heuristic evaluation function V(s) = w^T * φ(s).
All features are computed from a specified player's perspective.
"""

from typing import Dict, List, Tuple, Optional, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.game.game import Game

from src.utils.constants import BLUE, RED, PAWN, MASTER, BLUE_SHRINE, RED_SHRINE, BOARD_SIZE


class FeatureVector(NamedTuple):
    """Named tuple holding all 11 features."""
    material_diff_students: int
    my_master_alive: int
    opp_master_captured: int
    master_safety_balance: int
    legal_moves_diff: int
    capture_moves_diff: int
    master_temple_distance_diff: float
    student_progress_diff: float
    central_control_diff: int
    card_mobility_diff: int
    master_escape_options: int


class FeatureExtractor:
    """
    Extracts features from Onitama board positions.

    All features are computed from the perspective of the specified player.
    Positive values indicate advantage for the perspective player.
    """

    # Central 3x3 squares
    CENTRAL_SQUARES = frozenset((x, y) for x in range(1, 4) for y in range(1, 4))

    def __init__(self):
        """Initialize the feature extractor."""
        pass

    def extract(self, game: 'Game', perspective: int) -> FeatureVector:
        """
        Extract all features from the current game state.

        Args:
            game: The current game state
            perspective: Player perspective (BLUE=0 or RED=1)

        Returns:
            FeatureVector with all 11 features
        """
        me, opp = self._get_players(perspective)
        my_temple, enemy_temple = self._get_temples(perspective)
        board = game.get_board_state()

        # Phase 1: Board scan (one pass)
        (material_diff, my_master_alive, opp_master_captured,
         my_master_pos, opp_master_pos, my_student_positions,
         opp_student_positions) = self._compute_material_features(board, me, opp)

        # Phase 2: Positional features
        (master_dist_diff, student_progress_diff,
         central_control_diff) = self._compute_positional_features(
            board, me, opp, my_master_pos, opp_master_pos,
            my_student_positions, opp_student_positions,
            my_temple, enemy_temple)

        # Phase 3: Mobility features (computes legal moves for both players)
        (legal_diff, capture_diff, safety_balance,
         escape_options, card_mobility_diff) = self._compute_mobility_features(
            game, me, opp, board, my_master_pos, opp_master_pos)

        return FeatureVector(
            material_diff_students=material_diff,
            my_master_alive=my_master_alive,
            opp_master_captured=opp_master_captured,
            master_safety_balance=safety_balance,
            legal_moves_diff=legal_diff,
            capture_moves_diff=capture_diff,
            master_temple_distance_diff=master_dist_diff,
            student_progress_diff=student_progress_diff,
            central_control_diff=central_control_diff,
            card_mobility_diff=card_mobility_diff,
            master_escape_options=escape_options
        )

    def extract_as_array(self, game: 'Game', perspective: int) -> List[float]:
        """
        Extract features as a list suitable for dot product with weights.

        Args:
            game: The current game state
            perspective: Player perspective (BLUE=0 or RED=1)

        Returns:
            List of 11 floats
        """
        fv = self.extract(game, perspective)
        return list(fv)

    def evaluate(self, game: 'Game', perspective: int, weights: List[float]) -> float:
        """
        Compute V(s) = w^T * φ(s) for the given state.

        Args:
            game: The current game state
            perspective: Player perspective
            weights: List of 11 weights

        Returns:
            Scalar evaluation score
        """
        features = self.extract_as_array(game, perspective)
        return sum(w * f for w, f in zip(weights, features))

    def _get_players(self, perspective: int) -> Tuple[int, int]:
        """Return (my_player, opponent_player) based on perspective."""
        return (perspective, RED if perspective == BLUE else BLUE)

    def _get_temples(self, perspective: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return (my_temple, enemy_temple) based on perspective."""
        if perspective == BLUE:
            return (BLUE_SHRINE, RED_SHRINE)
        else:
            return (RED_SHRINE, BLUE_SHRINE)

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _compute_material_features(
        self,
        board: Dict[Tuple[int, int], Tuple[int, int]],
        me: int,
        opp: int
    ) -> Tuple[int, int, int, Optional[Tuple[int, int]], Optional[Tuple[int, int]],
               List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Scan board once to extract material-related features.

        Returns:
            Tuple of:
            - material_diff_students: my pawns - opponent pawns
            - my_master_alive: 1 if my master exists, 0 otherwise
            - opp_master_captured: 1 if opponent master is dead, 0 otherwise
            - my_master_pos: position of my master or None
            - opp_master_pos: position of opponent master or None
            - my_student_positions: list of my pawn positions
            - opp_student_positions: list of opponent pawn positions
        """
        my_students = 0
        opp_students = 0
        my_master_pos = None
        opp_master_pos = None
        my_student_positions = []
        opp_student_positions = []

        for pos, (player, piece_type) in board.items():
            if player == me:
                if piece_type == MASTER:
                    my_master_pos = pos
                else:
                    my_students += 1
                    my_student_positions.append(pos)
            else:
                if piece_type == MASTER:
                    opp_master_pos = pos
                else:
                    opp_students += 1
                    opp_student_positions.append(pos)

        return (
            my_students - opp_students,
            1 if my_master_pos is not None else 0,
            1 if opp_master_pos is None else 0,
            my_master_pos,
            opp_master_pos,
            my_student_positions,
            opp_student_positions
        )

    def _compute_positional_features(
        self,
        board: Dict[Tuple[int, int], Tuple[int, int]],
        me: int,
        opp: int,
        my_master_pos: Optional[Tuple[int, int]],
        opp_master_pos: Optional[Tuple[int, int]],
        my_student_positions: List[Tuple[int, int]],
        opp_student_positions: List[Tuple[int, int]],
        my_temple: Tuple[int, int],
        enemy_temple: Tuple[int, int]
    ) -> Tuple[float, float, int]:
        """
        Compute positional/progress features.

        Returns:
            Tuple of:
            - master_temple_distance_diff: opp master dist to my temple - my master dist to enemy temple
            - student_progress_diff: opp avg student dist to my temple - my avg to enemy temple
            - central_control_diff: my pieces in center - opponent pieces in center
        """
        # master_temple_distance_diff
        # Higher = better (opponent farther from winning, I'm closer)
        if my_master_pos is not None and opp_master_pos is not None:
            opp_dist = self._manhattan_distance(opp_master_pos, my_temple)
            my_dist = self._manhattan_distance(my_master_pos, enemy_temple)
            master_temple_distance_diff = float(opp_dist - my_dist)
        elif my_master_pos is not None:
            # Opponent master dead - large positive value
            master_temple_distance_diff = 10.0
        elif opp_master_pos is not None:
            # My master dead - large negative value
            master_temple_distance_diff = -10.0
        else:
            # Both masters dead (shouldn't happen normally)
            master_temple_distance_diff = 0.0

        # student_progress_diff
        # opp avg student dist to my temple minus my avg student dist to enemy temple
        # Higher = better (opponent students farther, mine closer)
        if my_student_positions:
            my_avg = sum(self._manhattan_distance(pos, enemy_temple)
                        for pos in my_student_positions) / len(my_student_positions)
        else:
            my_avg = 8.0  # Max distance if no students

        if opp_student_positions:
            opp_avg = sum(self._manhattan_distance(pos, my_temple)
                         for pos in opp_student_positions) / len(opp_student_positions)
        else:
            opp_avg = 8.0  # Max distance if no students

        student_progress_diff = opp_avg - my_avg

        # central_control_diff
        my_central = sum(1 for pos, (player, _) in board.items()
                        if player == me and pos in self.CENTRAL_SQUARES)
        opp_central = sum(1 for pos, (player, _) in board.items()
                         if player == opp and pos in self.CENTRAL_SQUARES)
        central_control_diff = my_central - opp_central

        return (master_temple_distance_diff, student_progress_diff, central_control_diff)

    def _compute_moves_for_player(
        self,
        game: 'Game',
        player: int
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """
        Compute legal moves for any player (not just current player).

        This is needed because game.get_legal_moves() only works for the current player.

        Args:
            game: The game state
            player: The player to compute moves for

        Returns:
            List of (from_pos, to_pos, card_name) tuples
        """
        legal_moves = []
        board = game.get_board_state()

        for card in game.get_player_cards(player):
            movements = card.get_movements(player)

            for pos, (piece_player, _) in board.items():
                if piece_player != player:
                    continue

                for dx, dy in movements:
                    to_pos = (pos[0] + dx, pos[1] + dy)
                    to_x, to_y = to_pos

                    # Check bounds
                    if not (0 <= to_x < BOARD_SIZE and 0 <= to_y < BOARD_SIZE):
                        continue

                    # Check not capturing own piece
                    if to_pos in board and board[to_pos][0] == player:
                        continue

                    legal_moves.append((pos, to_pos, card.name))

        return legal_moves

    def _compute_mobility_features(
        self,
        game: 'Game',
        me: int,
        opp: int,
        board: Dict[Tuple[int, int], Tuple[int, int]],
        my_master_pos: Optional[Tuple[int, int]],
        opp_master_pos: Optional[Tuple[int, int]]
    ) -> Tuple[int, int, int, int, int]:
        """
        Compute all mobility-related features.

        Returns:
            Tuple of:
            - legal_moves_diff: my legal moves - opponent legal moves
            - capture_moves_diff: my captures - opponent captures
            - master_safety_balance: my threats on opp master - opp threats on my master
            - master_escape_options: number of moves for my master
            - card_mobility_diff: total moves from my cards - opponent cards
        """
        my_moves = self._compute_moves_for_player(game, me)
        opp_moves = self._compute_moves_for_player(game, opp)

        # legal_moves_diff
        legal_moves_diff = len(my_moves) - len(opp_moves)

        # capture_moves_diff: moves where to_pos has opponent piece
        my_captures = sum(1 for m in my_moves if m[1] in board and board[m[1]][0] == opp)
        opp_captures = sum(1 for m in opp_moves if m[1] in board and board[m[1]][0] == me)
        capture_moves_diff = my_captures - opp_captures

        # master_safety_balance: my moves capturing opp master - opp moves capturing my master
        my_master_threats = 0
        opp_master_threats = 0

        if opp_master_pos is not None:
            my_master_threats = sum(1 for m in my_moves if m[1] == opp_master_pos)
        if my_master_pos is not None:
            opp_master_threats = sum(1 for m in opp_moves if m[1] == my_master_pos)

        master_safety_balance = my_master_threats - opp_master_threats

        # master_escape_options: my moves that move the master
        if my_master_pos is not None:
            master_escape_options = sum(1 for m in my_moves if m[0] == my_master_pos)
        else:
            master_escape_options = 0

        # card_mobility_diff: count of moves enabled by each player's cards
        # This is the same as legal_moves_diff since all moves come from cards
        # But we compute it explicitly for clarity
        card_mobility_diff = len(my_moves) - len(opp_moves)

        return (legal_moves_diff, capture_moves_diff, master_safety_balance,
                master_escape_options, card_mobility_diff)
