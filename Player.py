import copy
import gc
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import math
import requests
import pickle
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from Chinese_Chess_Game_Rules import ChessGame, PIECES, _Piece

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chess_training.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Từ điển giá trị quân cờ
PIECE_TO_NUMBER = {
    ('k', True): 9,    # Tướng đỏ: +9
    ('k', False): -9,  # Tướng đen: -9
    ('r', True): 7,    # Xe đỏ: +7
    ('r', False): -7,  # Xe đen: -7
    ('c', True): 5,    # Pháo đỏ: +5
    ('c', False): -5,  # Pháo đen: -5
    ('h', True): 4,    # Mã đỏ: +4
    ('h', False): -4,  # Mã đen: -4
    ('e', True): 3,    # Tượng đỏ: +3
    ('e', False): -3,  # Tượng đen: -3
    ('a', True): 2,    # Sĩ đỏ: +2
    ('a', False): -2,  # Sĩ đen: -2
    ('p', True): 1,    # Tốt đỏ: +1
    ('p', False): -1,  # Tốt đen: -1
}

# Mạng Policy (dự đoán xác suất nước đi)
class PolicyNetwork(nn.Module):
    def __init__(self, action_size=200):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 10 * 9, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, state):
        x = state.view(-1, 1, 10, 9)  # [batch, 1, 10, 9]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

# Mạng Value (đánh giá trạng thái)
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 10 * 9, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = state.view(-1, 1, 10, 9)  # [batch, 1, 10, 9]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output trong [-1, 1]

# Node cho MCTS (Monte Carlo Tree Search)
class MCTSNode:
    def __init__(self, game, parent=None, move=None, prior_p=1.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = {}  # move: child_node
        self.N = {}        # visit count
        self.W = {}        # total value
        self.Q = {}        # mean value
        self.P = {}        # prior probabilities
        self.visits = 0
        self.prior_p = prior_p
        self.lock = threading.Lock()

    def expand(self, moves, probs):
        with self.lock:
            for move, prob in zip(moves, probs):
                if move not in self.children:
                    new_game = self.game.copy_and_make_move(move)
                    new_game.board_history = {}
                    self.children[move] = MCTSNode(new_game, self, move, prob)
                    self.N[move] = 0
                    self.W[move] = 0
                    self.Q[move] = 0
                    self.P[move] = prob
            logging.debug(f"Expanded node with {len(self.children)} children")

    def select(self, c=1.5):
        with self.lock:
            best_score = -float('inf')
            best_move = None
            for move in self.P:
                u = self.P[move] * c * (np.sqrt(self.visits + 1) / (1 + self.N[move]))
                score = self.Q[move] + u
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_move

    def update(self, move, value):
        with self.lock:
            self.N[move] = self.N.get(move, 0) + 1
            self.W[move] = self.W.get(move, 0) + value
            self.Q[move] = self.W[move] / self.N[move]
            self.visits += 1

# Class chính điều khiển AI cờ tướng
class ChessAgent:
    def __init__(self, state_size=(10, 9), action_size=200):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.update_frequency = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.policy_network = PolicyNetwork(action_size).to(self.device)
        self.value_network = ValueNetwork().to(self.device)
        self.policy_target_network = PolicyNetwork(action_size).to(self.device)
        self.value_target_network = ValueNetwork().to(self.device)

        # Tải mô hình đã lưu (nếu có)
        model_path = "trained_models/chinese_chess.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                logging.info("Loaded checkpoint data")
                self.gamma = checkpoint.get("gamma")
                self.epsilon = checkpoint.get("epsilon")
                self.epsilon_min = checkpoint.get("epsilon_min")
                self.epsilon_decay = checkpoint.get("epsilon_decay")
                self.learning_rate = checkpoint.get("learning_rate")
                self.policy_network.load_state_dict(checkpoint["policy_network"])
                self.value_network.load_state_dict(checkpoint["value_network"])
        self.policy_target_network.load_state_dict(self.policy_network.state_dict())
        self.value_target_network.load_state_dict(self.value_network.state_dict())

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        # Cấu hình session requests với cơ chế retry
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def board_to_tensor(self, game):
        board = game.get_board()
        numeric_board = torch.zeros((10, 9), dtype=torch.float32)
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece:
                    key = (piece.kind, piece.is_red)
                    if key in PIECE_TO_NUMBER:
                        numeric_board[r, c] = PIECE_TO_NUMBER[key]
        return numeric_board.to(self.device)

    def encode_move(self, move, valid_moves):
        return valid_moves.index(move) if move in valid_moves else 0

    def board_to_score(self, game):
        board = game.get_board()
        score_board = np.zeros((10, 9), dtype=np.float32)
        turn = game.is_red_move()
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece:
                    if turn == piece.is_red:
                        score_board[r][c] = PIECE_TO_NUMBER[piece.kind]
                    else:
                        score_board[r][c] = -PIECE_TO_NUMBER[piece.kind]
        return score_board

    def board_to_fen(self, game):
        board = game.get_board()
        fen_rows = []
        
        for r in range(10):
            row = ""
            empty_count = 0
            for c in range(9):
                piece = board[r][c]
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row += str(empty_count)
                        empty_count = 0
                    kind = piece.kind
                    if piece.is_red:
                        row += kind.upper()
                    else:
                        row += kind.lower()
            if empty_count > 0:
                row += str(empty_count)
            fen_rows.append(row)
        
        board_fen = "/".join(fen_rows)
        side = "w" if game.is_red_move() else "b"
        move_count = str(game._move_count // 2)
        fen = f"{board_fen} {side} {move_count}"
        return fen

    def convert_cdb_move_to_game_format(self, cdb_move, game):
        """
        Chuyển đổi nước đi từ định dạng CDB (ví dụ: b6e6) sang định dạng game (ví dụ: r1+1).
        """
        if not cdb_move or len(cdb_move) != 4:
            return None
        
        from_pos, to_pos = cdb_move[:2], cdb_move[2:]
        
        from_col = ord(from_pos[0]) - ord('a')  # a=0, b=1, ..., i=8
        from_row = int(from_pos[1])  # 0-9
        to_col = ord(to_pos[0]) - ord('a')
        to_row = int(to_pos[1])

        # Kiểm tra tọa độ hợp lệ
        if not (0 <= from_col <= 8 and 0 <= from_row <= 9 and 0 <= to_col <= 8 and 0 <= to_row <= 9):
            logging.warning(f"Invalid coordinates in move {cdb_move}: from ({from_row}, {from_col}), to ({to_row}, {to_col})")
            return None

        # Chuyển đổi tọa độ từ hệ CDB sang hệ game
        from_col_game = 8 - from_col  # cột: a→8, b→7, ..., i→0
        from_row_game = 9 - from_row  # hàng: 0→9, 1→8, ..., 9→0
        to_col_game = 8 - to_col
        to_row_game = 9 - to_row

        # Lấy quân cờ tại vị trí ban đầu
        board = game.get_board()
        piece = board[from_row][from_col]
        if not piece:
            logging.warning(f"No piece found at position ({from_row}, {from_col}) for move {cdb_move}")
            return None
        
        piece_kind = piece.kind.lower()
        is_red = piece.is_red

        # Xác định nước đi trong định dạng game
        from_str = f"{piece_kind}{from_col_game + 1}"
        if to_row_game == from_row_game:  # Di chuyển ngang
            col_diff = to_col_game - from_col_game
            if col_diff > 0:
                move = f"{from_str}+{col_diff}"
            else:
                move = f"{from_str}{col_diff}"
        else:  # Di chuyển dọc
            # Với Pháo, cần kiểm tra xem có quân để nhảy qua không
            if piece_kind == 'c':
                direction = 1 if to_row_game > from_row_game else -1
                step = from_row + direction
                pieces_between = 0
                # Đảm bảo step không vượt quá giới hạn bàn cờ
                while 0 <= step <= 9 and step != to_row:
                    if board[step][from_col]:
                        pieces_between += 1
                    step += direction
                if pieces_between == 0 and not board[to_row][to_col]:  # Di chuyển thông thường
                    move = f"{from_str}.{to_col_game + 1}"
                elif pieces_between == 1 and board[to_row][to_col]:  # Ăn quân
                    move = f"{from_str}.{to_col_game + 1}"
                else:
                    logging.warning(f"Invalid cannon move {cdb_move}: {pieces_between} pieces between, target occupied: {bool(board[to_row][to_col])}")
                    return None
            else:
                move = f"{from_str}.{to_col_game + 1}"

        return move

    def convert_fen_move_to_game_move(self, fen_move, game):
        if len(fen_move) != 4:
            return None
        
        from_pos, to_pos = fen_move[:2], fen_move[2:]
        
        from_col = ord(from_pos[0]) - ord('a')  # a=0, b=1, ..., i=8
        from_row = int(from_pos[1])  # 0-9
        to_col = ord(to_pos[0]) - ord('a')
        to_row = int(to_pos[1])

        # Kiểm tra tọa độ hợp lệ
        if not (0 <= from_col <= 8 and 0 <= from_row <= 9 and 0 <= to_col <= 8 and 0 <= to_row <= 9):
            logging.warning(f"Invalid coordinates in move {fen_move}: from ({from_row}, {from_col}), to ({to_row}, {to_col})")
            return None

        from_col_game = 8 - from_col  # cột: a→8, b→7, ..., i→0
        from_row_game = 9 - from_row  # hàng: 0→9, 1→8, ..., 9→0
        to_col_game = 8 - to_col
        to_row_game = 9 - to_row

        # Lấy quân cờ tại vị trí ban đầu
        board = game.get_board()
        piece = board[from_row][from_col]
        if not piece:
            logging.warning(f"No piece found at position ({from_row}, {from_col}) for move {fen_move}")
            return None
        
        piece_kind = piece.kind.lower()
        is_red = piece.is_red

        # Xác định nước đi trong định dạng game
        from_str = f"{piece_kind}{from_col_game + 1}"
        if to_row_game == from_row_game:  # Di chuyển ngang
            col_diff = to_col_game - from_col_game
            if col_diff > 0:
                move = f"{from_str}+{col_diff}"
            else:
                move = f"{from_str}{col_diff}"
        else:  # Di chuyển dọc
            # Với Pháo, cần kiểm tra xem có quân để nhảy qua không
            if piece_kind == 'c':
                direction = 1 if to_row_game > from_row_game else -1
                step = from_row + direction
                pieces_between = 0
                # Đảm bảo step không vượt quá giới hạn bàn cờ
                while 0 <= step <= 9 and step != to_row:
                    if board[step][from_col]:
                        pieces_between += 1
                    step += direction
                if pieces_between == 0 and not board[to_row][to_col]:  # Di chuyển thông thường
                    move = f"{from_str}.{to_col_game + 1}"
                elif pieces_between == 1 and board[to_row][to_col]:  # Ăn quân
                    move = f"{from_str}.{to_col_game + 1}"
                else:
                    logging.warning(f"Invalid cannon move {fen_move}: {pieces_between} pieces between, target occupied: {bool(board[to_row][to_col])}")
                    return None
            else:
                move = f"{from_str}.{to_col_game + 1}"

        return move

    def apply_fen_with_moves(self, fen_with_moves, game):
        parts = fen_with_moves.split()
        fen = parts[0] + " " + parts[1]
        move_part = parts[2] if len(parts) > 2 else None
        move = parts[3] if len(parts) > 3 else None

        game.set_board_from_fen(fen)

        if move_part == "moves" and move:
            game_move = self.convert_fen_move_to_game_move(move, game)
            if game_move:
                valid_moves = game.get_valid_moves()
                if game_move in valid_moves:
                    game.make_move(game_move)
                    logging.info(f"Applied move: {move} (game format: {game_move})")
                else:
                    logging.warning(f"Move {game_move} is not valid for the given position")
            else:
                logging.warning(f"Could not convert move: {move}")

        new_fen = self.board_to_fen(game)
        return new_fen

    def fetch_moves_from_cdb(self, fen):
        """
        Gửi FEN đến chessdb.cn và lấy danh sách nước đi (queryall) cùng điểm đánh giá (queryscore).
        Returns: Tuple (moves, eval_score)
        - moves: List of (move, score, rank, winrate, count) tuples.
        - eval_score: Evaluation score of the position (or None if unavailable).
        """
        try:
            queryall_url = "http://www.chessdb.cn/chessdb.php"
            queryall_params = {
                "action": "queryall",
                "board": fen,
                "showall": 0,
                "learn": 1
            }
            response = self.session.get(queryall_url, params=queryall_params, timeout=10)
            response.raise_for_status()
            
            logging.debug(f"queryall response: {response.text}")
            print(f"API queryall response: {response.text}")
            
            moves = []
            if response.text in ["invalid board", "unknown", "checkmate", "stalemate"]:
                logging.warning(f"queryall response: {response.text}")
            else:
                move_entries = response.text.split("|")
                invalid_note_count = 0
                for entry in move_entries:
                    if not entry:
                        continue
                    fields = dict(field.split(":") for field in entry.split(","))
                    move = fields.get("move", "")
                    score_str = fields.get("score", "0")
                    try:
                        score_str = score_str.replace('\x00', '')
                        score = int(score_str)
                    except ValueError:
                        logging.warning(f"Invalid score format: {score_str}, setting score to 0")
                        score = 0
                    rank_str = fields.get("rank", "0")
                    try:
                        rank_str = rank_str.replace('\x00', '')
                        rank = int(rank_str)
                    except ValueError:
                        logging.warning(f"Invalid rank format: {rank_str}, setting rank to 0")
                        rank = 0
                    winrate_str = fields.get("winrate", "0.0")
                    try:
                        winrate_str = winrate_str.replace('\x00', '')
                        winrate = float(winrate_str) / 100.0  # Chuẩn hóa từ [0, 100] về [0, 1]
                        winrate = min(max(winrate, 0.0), 1.0)
                    except ValueError:
                        logging.warning(f"Invalid winrate format: {winrate_str}, setting winrate to 0.0")
                        winrate = 0.0
                    note = fields.get("note", "(0)")
                    try:
                        count_str = note.strip("()")
                        count = int(count_str)
                    except ValueError:
                        invalid_note_count += 1
                        count = 0
                    moves.append((move, score, rank, winrate, count))
                if invalid_note_count > 0:
                    logging.warning(f"Encountered {invalid_note_count} invalid note formats in this query, all counts set to 0")
            
            queryscore_params = {
                "action": "queryscore",
                "board": fen,
                "learn": 1
            }
            response = self.session.get(queryall_url, params=queryscore_params, timeout=10)
            response.raise_for_status()
            
            logging.debug(f"queryscore response: {response.text}")
            print(f"API queryscore response: {response.text}")
            
            eval_score = None
            if response.text.startswith("eval:"):
                eval_score_str = response.text.split(":")[1]
                try:
                    eval_score_str = eval_score_str.replace('\x00', '')
                    eval_score = int(eval_score_str) / 100.0
                except ValueError:
                    logging.warning(f"Invalid eval_score format: {eval_score_str}, setting eval_score to None")
                    eval_score = None
            elif response.text in ["invalid board", "unknown"]:
                logging.warning(f"queryscore response: {response.text}")
            
            time.sleep(2)
            return moves, eval_score
        except Exception as e:
            logging.error(f"Error fetching data from CDB: {e}")
            return [], None

    def calculate_reward(self, game, prev_board, curr_board):
        reward = 0.0
        winner = game.get_winner()
        if winner:
            if winner == 'Red' and game.is_red_move() or winner == 'Black' and not game.is_red_move():
                return 8.0
            elif winner == 'Draw':
                return 0.0
            else:
                return -8.0

        move_count = game._move_count
        decay_factor = np.exp(-0.002 * move_count)

        piece_types = ['k', 'r', 'c', 'h', 'e', 'a', 'p']
        curr_counts = {kind: {'red': 0, 'black': 0} for kind in piece_types}
        for r in range(10):
            for c in range(9):
                piece = curr_board[r][c]
                if piece:
                    side = 'red' if piece.is_red else 'black'
                    if piece.kind in piece_types:
                        curr_counts[piece.kind][side] += 1
                    else:
                        logging.warning(f"Invalid piece kind at ({r}, {c}): {piece.kind}")

        for kind in piece_types:
            if (kind, True) in PIECE_TO_NUMBER:
                value = abs(PIECE_TO_NUMBER[(kind, True)]) / 3.0
                if game.is_red_move():
                    if curr_counts[kind]['black'] == 0:
                        reward += value * decay_factor
                else:
                    if curr_counts[kind]['red'] == 0:
                        reward += value * decay_factor
            else:
                logging.warning(f"Key {(kind, True)} not found in PIECE_TO_NUMBER")

        if game.is_in_check(curr_board, not game.is_red_move()):
            reward += 1.0 * decay_factor
        return reward

    def save_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
        if len(self.replay_buffer) > 105000:
            self.replay_buffer.pop(0)

    def create_opening_move(self):
        game = ChessGame()
        game.board_history = {}
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        opening_moves = ["c2.5", "p3+1", "h2+3", "r1.2", "h8+9", "r2+6"] if game.is_red_move() else \
                        ["h8+7", "r9+1", "r9.4", "c8-1", "c8.5", "r4+6"]
        valid_opening_moves = [move for move in opening_moves if move in valid_moves]
        if not valid_opening_moves:
            move = random.choice(valid_moves)
        else:
            move = random.choice(valid_opening_moves)
        
        state = self.board_to_tensor(game)
        action_idx = self.encode_move(move, valid_moves)
        next_game = game.copy_and_make_move(move)
        next_state = self.board_to_tensor(next_game)
        reward = 8.0
        return (state, action_idx, reward, next_state)

    def set_training_mode(self, mode=True):
        def _set_module_training(module, mode):
            module.training = mode
            for name, child in module.named_children():
                _set_module_training(child, mode)
        _set_module_training(self.policy_network, mode)
        _set_module_training(self.value_network, mode)
        logging.debug(f"Set networks to {'training' if mode else 'evaluation'} mode")

    def train_main_network(self, batch_size=1024):
        print("Bắt đầu train")
        if len(self.replay_buffer) < batch_size:
            logging.info(f"Not enough samples in replay buffer: {len(self.replay_buffer)}/{batch_size}")
            return

        self_play = []
        batch_size = min(batch_size, len(self.replay_buffer))
        len_self_play = int(batch_size * 0.06)
        for _ in range(len_self_play):
            result = self.create_opening_move()
            if result:
                self_play.append(result)

        index = random.sample(range(len(self.replay_buffer)), batch_size - len_self_play)
        exp_replay = [self.replay_buffer[i] for i in index]
        exp_replay += self_play

        state_batch, action_batch, reward_batch, next_state_batch = zip(*exp_replay)
        state_batch = torch.stack(state_batch).to(self.device).float()
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)
        next_state_batch = torch.stack(next_state_batch).to(self.device).float()

        print(f"Replay buffer size: {len(self.replay_buffer)}")
        print(f"Reward stats: mean={reward_batch.mean():.4f}, std={reward_batch.std():.4f}")

        self.set_training_mode(True)

        self.policy_optimizer.zero_grad()
        policy_output = self.policy_network(state_batch)
        action_prob = policy_output.gather(1, action_batch.view(-1, 1)).squeeze()
        action_prob = torch.clamp(action_prob, 1e-10, 1.0)
        policy_loss = -torch.mean(torch.log(action_prob) * reward_batch)

        self.value_optimizer.zero_grad()
        value_output = self.value_network(state_batch).squeeze()
        value_loss = nn.MSELoss()(value_output, reward_batch)

        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()

        print(f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
        print(f"Total Loss: {total_loss.item():.4f}")

        if self.epsilon * self.epsilon_decay > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.update_frequency -= 1
        if self.update_frequency == 0:
            self.value_target_network.load_state_dict(self.value_network.state_dict())
            self.policy_target_network.load_state_dict(self.policy_network.state_dict())
            self.update_frequency = 10

        torch.cuda.empty_cache()
        gc.collect()

    def save(self):
        save_model = {
            "policy_network": self.policy_network.state_dict(),
            "value_network": self.value_network.state_dict(),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate
        }
        model_path = "trained_models/chinese_chess_alpha.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(save_model, model_path)
        logging.info(f"Saved model at {model_path}")

    def get_move(self, game, simulations=1000, c_puct=1.5):
        root = MCTSNode(game)
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        for _ in range(simulations):
            node = root
            sim_game = game.copy_and_make_move(None)
            sim_game.board_history = {}
            action_path = []

            while node.children:
                move = node.select(c=c_puct)
                if move is None:
                    break
                action_path.append((node, move))
                node = node.children[move]
                sim_game = sim_game.copy_and_make_move(move)
                sim_game.board_history = {}

            if not sim_game.get_winner():
                valid_moves = sim_game.get_valid_moves()
                if valid_moves:
                    state_tensor = self.board_to_tensor(sim_game)
                    with torch.no_grad():
                        probs = self.policy_target_network(state_tensor.unsqueeze(0)).cpu().numpy()[0]
                    probs = probs[:len(valid_moves)] / np.sum(probs[:len(valid_moves)])
                    node.expand(valid_moves, probs)
                    move = valid_moves[np.random.choice(len(valid_moves), p=probs)]
                    action_path.append((node, move))
                    node = node.children[move]
                    sim_game = sim_game.copy_and_make_move(move)
                    value = self.simulate(sim_game)
                else:
                    state_tensor = self.board_to_tensor(sim_game)
                    with torch.no_grad():
                        value = self.value_target_network(state_tensor.unsqueeze(0)).cpu().numpy()[0]
            else:
                winner = sim_game.get_winner()
                value = 1.0 if (winner == 'Red' and game.is_red_move()) or (winner == 'Black' and not game.is_red_move()) else -1.0

            for parent_node, move in action_path:
                parent_node.update(move, value)
            if not action_path:
                root.visits += 1

        best_move = max(root.N, key=root.N.get) if root.N else None
        for move in root.N:
            logging.info(f"Move: {move}, Visit count: {root.N[move]}")
        return best_move

    def simulate(self, game):
        sim_game = game.copy_and_make_move(None)
        sim_game.board_history = {}
        while not sim_game.get_winner():
            valid_moves = sim_game.get_valid_moves()
            if not valid_moves:
                break
            state_tensor = self.board_to_tensor(sim_game)
            with torch.no_grad():
                probs = self.policy_target_network(state_tensor.unsqueeze(0)).cpu().numpy()[0]
            probs = probs[:len(valid_moves)] / np.sum(probs[:len(valid_moves)])
            move = valid_moves[np.random.choice(len(valid_moves), p=probs)]
            sim_game = sim_game.copy_and_make_move(move)
            sim_game.board_history = {}
        
        winner = sim_game.get_winner()
        state_tensor = self.board_to_tensor(sim_game)
        with torch.no_grad():
            value = self.value_target_network(state_tensor.unsqueeze(0)).cpu().numpy()[0]
        if winner:
            return 1.0 if (winner == 'Red' and game.is_red_move()) or (winner == 'Black' and not game.is_red_move()) else -1.0
        return value

# Thêm phương thức set_board_from_fen vào ChessGame nếu chưa có
def set_board_from_fen(self, fen):
    parts = fen.split()
    board_fen = parts[0]
    side = parts[1]
    move_count = int(parts[2]) if len(parts) > 2 else 0

    self.board = [[None for _ in range(9)] for _ in range(10)]
    rows = board_fen.split('/')
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                is_red = char.isupper()
                kind = char.lower()
                self.board[r][c] = _Piece(kind, is_red)
                c += 1

    self._red_to_move = (side == 'w')
    self._move_count = move_count * 2

if not hasattr(ChessGame, 'set_board_from_fen'):
    ChessGame.set_board_from_fen = set_board_from_fen

# Vòng lặp huấn luyện chính
if __name__ == "__main__":
    env = ChessGame()
    env.board_history = {}
    state_size = (10, 9)
    action_size = 200
    n_timesteps = 500
    batch_size = 1024

    agent = ChessAgent(state_size=state_size, action_size=action_size)
    start_ep = 0
    dataset = []

    try:
        with open("cdb_dataset_final.pkl", "rb") as f:
            dataset = pickle.load(f)
        logging.info(f"Loaded dataset with {len(dataset)} samples")
        print(f"Loaded dataset with {len(dataset)} samples")
    except FileNotFoundError:
        dataset = []

    # Chuỗi FEN với nước đi
    fen_with_moves = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w moves b2d2"

    # Áp dụng FEN và nước đi
    new_fen = agent.apply_fen_with_moves(fen_with_moves, env)
    print(f"New FEN after move: {new_fen}")

    state = agent.board_to_tensor(env)
    prev_board = env.get_board()

    ep = start_ep
    try:
        while True:
            ep += 1
            ep_rewards = 0
            logging.info(f"\n=== Starting Episode {ep} ===")
            print(f"\n=== Starting Episode {ep} ===")

            for t in range(n_timesteps):
                action = agent.get_move(env)
                if action is None:
                    logging.info("No valid moves available!")
                    print("No valid moves available!")
                    break
                
                env.make_move(action)
                env.__str__()
                next_state = agent.board_to_tensor(env)
                curr_board = env.get_board()
                reward = agent.calculate_reward(env, prev_board, curr_board)
                
                action_idx = agent.encode_move(action, env.get_valid_moves())
                agent.save_experience(state, action_idx, reward, next_state)

                fen = agent.board_to_fen(env)
                moves_from_db, eval_score = agent.fetch_moves_from_cdb(fen)
                if moves_from_db:
                    print(f"FEN: {fen}")
                    valid_moves = env.get_valid_moves()
                    converted_moves = []
                    for cdb_move, score, rank, winrate, count in moves_from_db:
                        game_move = agent.convert_cdb_move_to_game_format(cdb_move, env)
                        if game_move and game_move in valid_moves:
                            converted_moves.append((game_move, score, rank, winrate, count))
                        else:
                            logging.warning(f"Move {cdb_move} from CDB is not valid in current game state")
                    print(f"Dataset from CDB: {converted_moves}")
                    print(f"Evaluation score: {eval_score}")
                    for move, score, rank, winrate, count in converted_moves:
                        move_idx = agent.encode_move(move, valid_moves)
                        temp_game = env.copy_and_make_move(move)
                        next_state_temp = agent.board_to_tensor(temp_game)
                        db_reward = (score + winrate * 2) / 3.0
                        if eval_score is not None:
                            db_reward = (db_reward + eval_score) / 2.0
                        db_reward += math.log(count + 1) * 0.1
                        dataset.append((next_state, move_idx, db_reward, next_state_temp))
                
                state = next_state
                prev_board = curr_board
                ep_rewards += reward

                logging.info(f"Move: {action} | Reward: {reward:.4f} | Move count: {env._move_count}")
                print(f"Move: {action} | Reward: {reward:.4f} | Move count: {env._move_count}")

                if env.get_winner():
                    winner = env.get_winner()
                    logging.info(f"Episode {ep} ended with winner: {winner}, Total reward: {ep_rewards:.4f}")
                    print(f"Episode {ep} ended with winner: {winner}, Total reward: {ep_rewards:.4f}")
                    break

                if len(dataset) >= batch_size:
                    random.shuffle(dataset)
                    for i in range(0, len(dataset), batch_size):
                        batch = dataset[i:i + batch_size]
                        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)
                        state_batch = torch.stack(state_batch).to(agent.device).float()
                        action_batch = torch.tensor(action_batch, dtype=torch.long).to(agent.device)
                        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(agent.device)
                        next_state_batch = torch.stack(next_state_batch).to(agent.device).float()

                        agent.set_training_mode(True)
                        agent.policy_optimizer.zero_grad()
                        policy_output = agent.policy_network(state_batch)
                        action_prob = policy_output.gather(1, action_batch.view(-1, 1)).squeeze()
                        action_prob = torch.clamp(action_prob, 1e-10, 1.0)
                        policy_loss = -torch.mean(torch.log(action_prob) * reward_batch)

                        agent.value_optimizer.zero_grad()
                        value_output = agent.value_network(state_batch).squeeze()
                        value_loss = nn.MSELoss()(value_output, reward_batch)

                        total_loss = policy_loss + value_loss
                        total_loss.backward()
                        agent.policy_optimizer.step()
                        agent.value_optimizer.step()

                        print(f"Training with CDB dataset - Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
                    dataset = []

                if len(agent.replay_buffer) >= batch_size:
                    agent.train_main_network(batch_size)

            if not env.get_winner():
                logging.info(f"Episode {ep} reached {t + 1} moves, Total reward: {ep_rewards:.4f}")
                print(f"Episode {ep} reached {t + 1} moves, Total reward: {ep_rewards:.4f}")

            if ep % 5 == 0:
                if dataset:
                    with open(f"cdb_dataset_ep_{ep}.pkl", "wb") as f:
                        pickle.dump(dataset, f)
                    logging.info(f"Saved CDB dataset at episode {ep}")
                    print(f"Saved CDB dataset at episode {ep}")

            if ep % 10 == 0:
                agent.save()
                logging.info(f"Saved model at episode {ep}")
                print(f"Saved model at episode {ep}")

    except KeyboardInterrupt:
        logging.info(f"Training interrupted at episode {ep}. Saving model...")
        print(f"Training interrupted at episode {ep}. Saving model...")
        agent.save()
        logging.info(f"Saved model at episode {ep}")
        print(f"Saved model at episode {ep}")
        if dataset:
            with open(f"cdb_dataset_final.pkl", "wb") as f:
                pickle.dump(dataset, f)
            logging.info("Saved final CDB dataset")
            print("Saved final CDB dataset")
        raise