import random
import math
from Chinese_Chess_Game_Rules import ChessGame, PIECES, calculate_absolute_points, _Piece, _get_index_movement
import numpy as np
from collections import deque

class Player:
    def make_move(self, game, previous_move):
        raise NotImplementedError("Phải triển khai make_move trong lớp con")

    def reload_tree(self):
        raise NotImplementedError("Phải triển khai reload_tree trong lớp con")

class RandomPlayer(Player):
    def make_move(self, game, previous_move):
        valid_moves = game.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None

    def reload_tree(self):
        pass

class MCTSNode:
    def __init__(self, game, move=None, parent=None):
        self.game = game
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = 1.0

    def ucb1(self, exploration_weight=1.0):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self):
        valid_moves = self.game.get_valid_moves()
        for move in valid_moves:
            new_game = self.game.copy_and_make_move(move)
            self.children.append(MCTSNode(new_game, move, self))
        return self.children[0] if self.children else None

    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(-value)

class MCTSPlayer(Player):
    def __init__(self, iterations=1000, exploration_weight=1.0):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.move_history = deque(maxlen=50)
        self.experience_buffer = deque(maxlen=10000)
        self.opening_moves = {
            'Red': [
                "c2.5", "h2+3", "r1+1", "h8+7", "r9+1",
                "c8+2", "p3+1", "e3+5", "r9.8", "p7+1",
                "h2+1", "r1.2"
            ],
            'Black': [
                "h8+7", "c8.5", "r9+1", "h2+3", "r1+1",
                "c2+2", "p7+1", "e7+5", "r1.2", "p3+1",
                "h8+9", "r9.8"
            ]
        }

    def board_to_tensor(self, game):
        board = game.get_board()
        tensor = np.zeros((10, 9, len(PIECES)), dtype=np.float32)
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece:
                    key = (piece.kind, piece.is_red)
                    if key in PIECES:
                        tensor[r, c, list(PIECES.keys()).index(key)] = 1
        return tensor

    def default_evaluate_board(self, game):
        board = game.get_board()
        total_score = calculate_absolute_points(board)

        # Thưởng khi kiểm soát trung tâm
        for y in range(4, 6):
            for x in range(3, 6):
                piece = board[y][x]
                if piece and piece.is_red == game.is_red_move():
                    total_score += 50 if piece.kind in {'r', 'c'} else 30 if piece.kind == 'h' else 10

        # Phạt nếu bị chiếu
        if game.is_in_check(board, game.is_red_move()):
            total_score -= 200

        # Thưởng nếu pháo có "ngòi" để tấn công
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece and piece.kind == 'c' and piece.is_red == game.is_red_move():
                    for direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        screen_count = 0
                        for i in range(1, 10):
                            ny, nx = y + direction[0] * i, x + direction[1] * i
                            if not (0 <= ny < 10 and 0 <= nx < 9):
                                break
                            if board[ny][nx]:
                                screen_count += 1
                            if screen_count == 1 and i < 9:
                                total_score += 20
                                break

        # Phạt mạnh nếu nước đi lặp lại
        current_move = self.move_history[-1] if self.move_history else None
        if current_move and list(self.move_history).count(current_move) >= 2:
            total_score -= 500  # Tăng mức phạt để tránh lặp lại

        # Thưởng nếu xe hoặc pháo di chuyển đến gần cung đối phương
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece and piece.is_red == game.is_red_move() and piece.kind in {'r', 'c'}:
                    if piece.is_red and y <= 2:  # Xe/pháo đỏ đến gần cung đen
                        total_score += 100
                    elif not piece.is_red and y >= 7:  # Xe/pháo đen đến gần cung đỏ
                        total_score -= 100

        # Thưởng nếu có cơ hội ăn quân
        for move in game.get_valid_moves():
            next_game = game.copy_and_make_move(move)
            y, x = _get_index_movement(board, move, game.is_red_move())
            captured = board[y][x]
            if captured and captured.is_red != game.is_red_move():
                total_score += 100 if captured.kind in {'r', 'c', 'h'} else 50

        # Chuẩn hóa điểm số
        total_score = total_score / 10000 * (1 if game.is_red_move() else -1)
        return min(max(total_score, -1), 1)

    def make_move(self, game, previous_move):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        move_count = len(self.move_history)
        if move_count < len(self.opening_moves['Red']):
            side = 'Red' if game.is_red_move() else 'Black'
            opening_move = self.opening_moves[side][move_count]
            if opening_move in valid_moves:
                self.move_history.append(opening_move)
                return opening_move

        root = MCTSNode(game)
        for _ in range(self.iterations):
            node = root
            while node.children and node.game.get_winner() is None:
                node = max(node.children, key=lambda c: c.ucb1(self.exploration_weight))
            if node.game.get_winner() is None:
                node = node.expand()
            if node:
                value = self.default_evaluate_board(node.game)
                node.backpropagate(value)
                self.experience_buffer.append((
                    self.board_to_tensor(node.parent.game), node.move, value,
                    self.board_to_tensor(node.game), node.game.get_winner()))
            else:
                break

        if not root.children:
            return random.choice(valid_moves) if valid_moves else None

        capturing_moves = []
        board = game.get_board()
        for child in root.children:
            y, x = _get_index_movement(board, child.move, game.is_red_move())
            captured = board[y][x]
            if captured and captured.is_red != game.is_red_move():
                capturing_moves.append((child.move, child.visits))

        if capturing_moves:
            best_move = max(capturing_moves, key=lambda x: x[1])[0]
        else:
            best_move = max(root.children, key=lambda c: c.visits).move

        self.move_history.append(best_move)
        return best_move

    def reload_tree(self):
        pass

if __name__ == "__main__":
    mcts_player = MCTSPlayer(iterations=500)

    game = ChessGame()
    print("\n=== Ván thử nghiệm ===")
    print(game)

    while game._move_count < 100:  # Chỉ dừng khi đạt 50 nước
        move = mcts_player.make_move(game, game.last_move)
        if move is None:
            print("Không còn nước đi hợp lệ!")
            break
        print(f"Nước đi: {move}")
        game.make_move(move)
        print(game)
        winner = game.get_winner()
        print(f"Debug: get_winner() = {winner}, move_count = {game._move_count}")
        if winner == 'Red' or winner == 'Black':
            print(f"{winner} wins!")
            break
        elif winner == 'Draw':
            print("Draw!")
            break

    winner = game.get_winner()
    print(f"Kết quả ván thử nghiệm: {winner if winner else 'Chưa có người thắng'}")