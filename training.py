import logging
from AlphaBetaAgent import AlphaBetaAgent, tree_to_xml, xml_to_tree, GameTree  # Added xml_to_tree
from Chinese_Chess_Game_Rules import ChessGame, calculate_absolute_points

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def train_alphabeta(xml_file: str, iterations: int, max_depth: int = 3) -> None:
    """
    Train AlphaBetaAgent by simulating games and expanding the GameTree.

    Args:
        xml_file (str): Path to the XML file to store the GameTree.
        iterations (int): Number of games to simulate.
        max_depth (int): Maximum depth for AlphaBetaAgent's search.
    """
    # Initialize agents
    red_agent = AlphaBetaAgent(max_depth=max_depth, xml_file=xml_file)
    black_agent = AlphaBetaAgent(max_depth=max_depth, xml_file=xml_file)

    # Ensure tree.xml exists with an empty tree if not present
    try:
        red_agent._game_tree = xml_to_tree(xml_file)
    except FileNotFoundError:
        logging.info(f"Creating new empty tree at {xml_file}")
        empty_tree = GameTree()
        tree_to_xml(empty_tree, xml_file)
        red_agent._game_tree = empty_tree
        black_agent._game_tree = empty_tree

    for i in range(iterations):
        logging.info(f"Starting simulation {i + 1}/{iterations}")
        game = ChessGame()
        moves = []
        points = []
        current_player = red_agent
        previous_move = None

        # Simulate a game
        while game.get_winner() is None:
            move = current_player.get_move(game)
            if move is None:
                logging.warning("No valid move available, ending game")
                break

            game.make_move(move)
            moves.append(move)
            points.append(calculate_absolute_points(game.get_board()))

            # Print board for debugging
            current_player.print_board(game)

            previous_move = move
            current_player = black_agent if current_player == red_agent else red_agent

        # Determine winner and assign probabilities
        winner = game.get_winner()
        red_win_prob = 1.0 if winner == 'Red' else 0.0
        black_win_prob = 1.0 if winner == 'Black' else 0.0

        logging.info(f"Game {i + 1} ended. Winner: {winner}. Moves: {len(moves)}")

        # Insert move sequence into both agents' trees
        red_agent._current_tree.insert_move_sequence(moves, points, red_win_prob, black_win_prob)
        black_agent._current_tree.insert_move_sequence(moves, points, red_win_prob, black_win_prob)

        # Store the updated trees
        red_agent.store_tree()
        black_agent.store_tree()

        # Reload trees for the next iteration
        red_agent._game_tree = xml_to_tree(xml_file)
        black_agent._game_tree = xml_to_tree(xml_file)
        red_agent._current_tree = GameTree()
        red_agent._current_subtree = red_agent._current_tree
        black_agent._current_tree = GameTree()
        black_agent._current_subtree = black_agent._current_tree

if __name__ == '__main__':
    # Example usage
    train_alphabeta('tree.xml', iterations=10, max_depth=2)