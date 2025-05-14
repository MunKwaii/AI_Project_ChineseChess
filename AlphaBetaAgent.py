import random
import logging
import xml.etree.cElementTree as ET
import math
from typing import Optional
from Chinese_Chess_Game_Rules import ChessGame, calculate_absolute_points, _get_wxf_movement, _wxf_to_index, _get_index_movement

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chess_minimax.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Constants from CSC111
GAME_START_MOVE = '*'
ESTIMATION = 0.8

class GameTree:
    """A decision tree for chess moves, adapted from CSC111 game_tree.py."""
    def __init__(self, move: str = GAME_START_MOVE,
                 is_red_move: bool = True, relative_points: int = 0,
                 red_win_probability: float = 0.0, black_win_probability: float = 0.0) -> None:
        self.move = move
        self.is_red_move = is_red_move
        self.red_win_probability = red_win_probability
        self.black_win_probability = black_win_probability
        self.relative_points = relative_points
        self._subtrees = []

    def get_subtrees(self) -> list['GameTree']:
        return self._subtrees

    def find_subtree_by_move(self, move: str) -> Optional['GameTree']:
        for subtree in self._subtrees:
            if subtree.move == move:
                return subtree
        return None

    def add_subtree(self, subtree: 'GameTree') -> None:
        self._subtrees.append(subtree)
        self._update_win_probabilities()

    def clean_subtrees(self) -> None:
        self._subtrees = []

    def insert_move_sequence(self, moves: list[str], points: list[int],
                             red_win_probability: float = 0.0,
                             black_win_probability: float = 0.0) -> None:
        self.insert_move_index(0, moves, points, red_win_probability, black_win_probability)

    def insert_move_index(self, curr_index: int, moves: list[str], points: list[int],
                          red_win_probability: float, black_win_probability: float) -> None:
        if curr_index == len(moves):
            return
        curr_move = moves[curr_index]
        relative_point = points[curr_index]
        for subtree in self._subtrees:
            if subtree.move == curr_move:
                subtree.insert_move_index(curr_index + 1, moves, points,
                                          red_win_probability, black_win_probability)
                self._update_win_probabilities()
                return
        if self.is_red_move:
            self.add_subtree(GameTree(move=curr_move, is_red_move=False,
                                      relative_points=relative_point,
                                      red_win_probability=red_win_probability,
                                      black_win_probability=black_win_probability))
        else:
            self.add_subtree(GameTree(move=curr_move, is_red_move=True,
                                      relative_points=relative_point,
                                      red_win_probability=red_win_probability,
                                      black_win_probability=black_win_probability))
        self._subtrees[-1].insert_move_index(curr_index + 1, moves, points,
                                             red_win_probability, black_win_probability)
        self._update_win_probabilities()

    def _update_win_probabilities(self) -> None:
        if self._subtrees == []:
            return
        subtrees_win_prob_red = [subtree.red_win_probability for subtree in self._subtrees]
        subtrees_win_prob_black = [subtree.black_win_probability for subtree in self._subtrees]
        if self.is_red_move:
            self.red_win_probability = max(subtrees_win_prob_red)
            half_len = math.ceil(len(subtrees_win_prob_black) * ESTIMATION)
            top_chances = sorted(subtrees_win_prob_black, reverse=True)[:half_len]
            self.black_win_probability = sum(top_chances) / half_len if half_len > 0 else 0.0
        else:
            self.black_win_probability = max(subtrees_win_prob_black)
            half_len = math.ceil(len(subtrees_win_prob_red) * ESTIMATION)
            top_chances = sorted(subtrees_win_prob_red, reverse=True)[:half_len]
            self.red_win_probability = sum(top_chances) / half_len if half_len > 0 else 0.0

    def merge_with(self, other_tree: 'GameTree') -> None:
        assert self.move == other_tree.move
        subtrees_moves = [sub.move for sub in self._subtrees]
        for subtree in other_tree.get_subtrees():
            if subtree.move in subtrees_moves:
                index = subtrees_moves.index(subtree.move)
                self._subtrees[index].merge_with(subtree)
            else:
                self.add_subtree(subtree)
        self.reevaluate()

    def reevaluate(self) -> None:
        self.purge()
        if self._subtrees == []:
            return
        for subtree in self._subtrees:
            subtree.reevaluate()
            if subtree._subtrees != [] and subtree.is_red_move:
                subtree._update_win_probabilities()
                subtree.relative_points = max(s.relative_points for s in subtree._subtrees)
            elif subtree._subtrees != [] and not subtree.is_red_move:
                subtree._update_win_probabilities()
                subtree.relative_points = min(s.relative_points for s in subtree._subtrees)

    def purge(self) -> None:
        moves_so_far = []
        for subtree in self._subtrees[:]:
            if subtree.move in moves_so_far:
                self._subtrees.remove(subtree)
                self.find_subtree_by_move(subtree.move).merge_with(subtree)
            else:
                moves_so_far.append(subtree.move)

def tree_to_xml(tree: GameTree, filename: str) -> None:
    bool_dict = {True: 't', False: 'f'}
    root = ET.Element('GameTree')
    root_move = ET.SubElement(root, 'm', m=str(tree.move), i=bool_dict[tree.is_red_move],
                              p=str(tree.relative_points),
                              r=str(tree.red_win_probability), b=str(tree.black_win_probability))
    _build_e_tree(root_move, tree)
    xml_tree = ET.ElementTree(root)
    xml_tree.write(filename)

def _build_e_tree(root_move: ET.Element, tree: GameTree) -> None:
    bool_dict = {True: 't', False: 'f'}
    for subtree in tree.get_subtrees():
        move = ET.SubElement(root_move, 'm', m=str(subtree.move),
                             i=bool_dict[subtree.is_red_move],
                             p=str(subtree.relative_points),
                             r=str(subtree.red_win_probability),
                             b=str(subtree.black_win_probability))
        _build_e_tree(move, subtree)

def xml_to_tree(filename: str) -> GameTree:
    tree = GameTree()
    try:
        with open(filename) as file:
            e_tree = ET.parse(file)
            root_move = e_tree.getroot()[0]
            _build_game_tree(root_move, tree)
    except FileNotFoundError:
        logging.warning(f"File {filename} not found, initializing empty GameTree")
        return GameTree()
    return tree

def _build_game_tree(move: ET.Element, tree: GameTree) -> None:
    tree.move = move.attrib['m']
    tree.is_red_move = move.attrib['i'] == 't'
    tree.relative_points = int(move.attrib['p'])
    tree.red_win_probability = float(move.attrib['r'])
    tree.black_win_probability = float(move.attrib['b'])
    for child_move in move:
        subtree = GameTree()
        _build_game_tree(child_move, subtree)
        tree.add_subtree(subtree)

class AlphaBetaAgent:
    def __init__(self, max_depth=3, xml_file='tree.xml'):
        """
        Initialize the Alpha-Beta Agent with GameTree support.
        
        Args:
            max_depth (int): Maximum depth for Minimax search.
            xml_file (str): Path to the XML file storing the GameTree.
        """
        self.max_depth = max_depth
        self.xml_file = xml_file
        self._game_tree = xml_to_tree(xml_file)
        self._current_tree = GameTree()
        self._current_subtree = self._current_tree
        logging.info(f"Initialized AlphaBetaAgent with max_depth={max_depth}, xml_file={xml_file}")

    def get_move(self, game, simulations=None, c_puct=None, max_depth=None, policy_threshold=None):
        """
        Select the best move using GameTree and Minimax with Alpha-Beta Pruning.
        
        Args:
            game (ChessGame): Current game state.
            simulations, c_puct, policy_threshold: Ignored (for compatibility).
            max_depth: Override max_depth if provided.
            
        Returns:
            str: Best move in WXF notation.
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            logging.info("No valid moves available!")
            return None

        depth = max_depth if max_depth is not None else self.max_depth
        candidate_moves = []
        best_score = -float('inf') if game.is_red_move() else float('inf')

        # Check if there are moves in the GameTree
        if self._game_tree and self._game_tree.get_subtrees():
            logging.info("Using GameTree for move selection")
            subtrees = self._game_tree.get_subtrees()
            valid_subtrees = [s for s in subtrees if s.move in valid_moves]
            
            if valid_subtrees:
                # Select moves with highest win probability
                if game.is_red_move():
                    max_prob = max(s.red_win_probability for s in valid_subtrees)
                    candidate_subtrees = [s for s in valid_subtrees if s.red_win_probability == max_prob]
                    min_opponent_prob = min(s.black_win_probability for s in candidate_subtrees)
                    candidate_subtrees = [s for s in candidate_subtrees if s.black_win_probability == min_opponent_prob]
                else:
                    max_prob = max(s.black_win_probability for s in valid_subtrees)
                    candidate_subtrees = [s for s in valid_subtrees if s.black_win_probability == max_prob]
                    min_opponent_prob = min(s.red_win_probability for s in candidate_subtrees)
                    candidate_subtrees = [s for s in candidate_subtrees if s.red_win_probability == min_opponent_prob]
                
                candidate_moves = [s.move for s in candidate_subtrees]
                logging.info(f"Found {len(candidate_moves)} candidate moves from GameTree: {candidate_moves}")

        # If no suitable moves from GameTree, use Minimax
        if not candidate_moves:
            logging.info("No suitable moves in GameTree, falling back to Minimax")
            for move in valid_moves:
                new_game = game.copy_and_make_move(move)
                score = self.alpha_beta(new_game, depth - 1, -float('inf'), float('inf'), not game.is_red_move())
                if game.is_red_move():
                    if score > best_score:
                        best_score = score
                        candidate_moves = [move]
                    elif score == best_score:
                        candidate_moves.append(move)
                else:
                    if score < best_score:
                        best_score = score
                        candidate_moves = [move]
                    elif score == best_score:
                        candidate_moves.append(move)
        
        # Randomly select one of the best moves
        selected_move = random.choice(candidate_moves) if candidate_moves else random.choice(valid_moves)
        
        # Update current tree
        new_subtree = GameTree(selected_move, not game.is_red_move())
        self._current_subtree.add_subtree(new_subtree)
        self._current_subtree = new_subtree
        self._game_tree = self._game_tree.find_subtree_by_move(selected_move) if self._game_tree else None
        
        logging.info(f"Selected move: {selected_move}, Score: {best_score}")
        return selected_move

    def alpha_beta(self, game, depth, alpha, beta, maximizing_player):
        """
        Minimax with Alpha-Beta Pruning.
        
        Args:
            game (ChessGame): Current game state.
            depth (int): Remaining depth to search.
            alpha (float): Best score for maximizer.
            beta (float): Best score for minimizer.
            maximizing_player (bool): True if current player is maximizer (Red).
            
        Returns:
            float: Best score for the current position.
        """
        winner = game.get_winner()
        if winner is not None:
            if winner == 'Red':
                return 10000
            elif winner == 'Black':
                return -10000
            else:
                return 0
        if depth == 0:
            score = calculate_absolute_points(game.get_board())
            if game.is_in_check(game.get_board(), game.is_red_move()):
                score -= 500 if game.is_red_move() else -500
            return score

        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return 0

        if maximizing_player:
            max_score = -float('inf')
            for move in valid_moves:
                new_game = game.copy_and_make_move(move)
                score = self.alpha_beta(new_game, depth - 1, alpha, beta, False)
                max_score = max(max_score, score)
                alpha = max(alpha, max_score)
                if beta <= alpha:
                    logging.debug(f"Pruned at depth {depth} for move {move}")
                    break
            return max_score
        else:
            min_score = float('inf')
            for move in valid_moves:
                new_game = game.copy_and_make_move(move)
                score = self.alpha_beta(new_game, depth - 1, alpha, beta, True)
                min_score = min(min_score, score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    logging.debug(f"Pruned at depth {depth} for move {move}")
                    break
            return min_score

    def store_tree(self):
        """
        Merge the current tree with the stored GameTree and save to xml_file.
        """
        logging.info("Storing GameTree to XML...")
        self._game_tree = xml_to_tree(self.xml_file)  # Reload to avoid overwrite issues
        self._game_tree.merge_with(self._current_tree)
        tree_to_xml(self._game_tree, self.xml_file)
        self._current_tree = GameTree()
        self._current_subtree = self._current_tree
        logging.info(f"Successfully stored GameTree to {self.xml_file}")

    def print_board(self, game):
        """
        Print the current board state (for debugging).
        
        Args:
            game (ChessGame): Current game state.
        """
        from Chinese_Chess_Game_Rules import print_board
        print_board(game.get_board())