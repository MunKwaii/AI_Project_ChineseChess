import copy
import gc
import os
import random
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from Chinese_Chess_Game_Rules import ChessGame, _Piece, PIECES, print_board, _wxf_to_index, _get_index_movement

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chess_training.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Định nghĩa giá trị quân cờ
PIECE_TO_NUMBER = {
    ('k', True): 9, ('k', False): -9,
    ('r', True): 7, ('r', False): -7,
    ('c', True): 5, ('c', False): -5,
    ('h', True): 4, ('h', False): -4,
    ('e', True): 3, ('e', False): -3,
    ('a', True): 2, ('a', False): -2,
    ('p', True): 1, ('p', False): -1,
}

# Mạng Policy
class PolicyNetwork(nn.Module):
    def __init__(self, action_size=4000):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 10 * 9, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, state):
        x = state.view(-1, 1, 10, 9)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)

# Mạng Value
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 10 * 9, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = state.view(-1, 1, 10, 9)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Nút MCTS
class MCTSNode:
    def __init__(self, game, parent=None, move=None, prior_p=1.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = {}
        self.N = {}
        self.W = {}
        self.Q = {}
        self.P = {}
        self.visits = 0
        self.prior_p = prior_p

    def expand(self, moves, probs):
        for move, prob in zip(moves, probs):
            if move not in self.children:
                new_game = self.game.copy_and_make_move(move)
                self.children[move] = MCTSNode(new_game, self, move, prob)
                self.N[move] = 0
                self.W[move] = 0
                self.Q[move] = 0
                self.P[move] = prob
        logging.debug(f"Expanded node with {len(self.children)} children")

    def select(self, c=2.0):
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
        self.N[move] = self.N.get(move, 0) + 1
        self.W[move] = self.W.get(move, 0) + value
        self.Q[move] = self.W[move] / self.N[move]
        self.visits += 1

# Lớp ChessAgent
class ChessAgent:
    def __init__(self, state_size=(10, 9), action_size=4000):
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
        if self.device.type == "cpu" and torch.cuda.is_available():
            logging.warning("GPU is available but not used. Consider enabling CUDA for faster training.")

        self.policy_network = PolicyNetwork(action_size).to(self.device)
        self.value_network = ValueNetwork().to(self.device)
        self.policy_target_network = PolicyNetwork(action_size).to(self.device)
        self.value_target_network = ValueNetwork().to(self.device)

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

        self.session = requests.Session()
        retries = Retry(total=10, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

        # Cache cho API
        self.api_cache = {}
        self.cache_path = "api_cache.pkl"
        self.load_api_cache()
        self.api_fail_count = 0
        self.api_success_count = 0
        self.api_fail_threshold = 10

        # Danh sách nước đi khai cuộc cố định
        self.opening_moves = [
            'c3+2', 'h2+3', 'r1+2', 'r9+2', 'p7+1', 'p3+1',
            'h8+7', 'c8+2', 'r1+1', 'r9+1', 'e3+5', 'e7+5'
        ]

        self.move_history = []

    def load_api_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self.api_cache = pickle.load(f)
                logging.info(f"Loaded API cache with {len(self.api_cache)} entries")
            except Exception as e:
                logging.error(f"Failed to load API cache: {e}")
                self.api_cache = {}

    def save_api_cache(self):
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.api_cache, f)
            logging.info(f"Saved API cache with {len(self.api_cache)} entries")
        except Exception as e:
            logging.error(f"Failed to save API cache: {e}")

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
        if move not in valid_moves:
            logging.warning(f"Move {move} not in valid moves, encoding as 0")
            return 0
        return valid_moves.index(move)

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
        logging.debug(f"Generated FEN: {fen}")
        return fen

    def validate_fen(self, fen):
        try:
            parts = fen.split()
            if len(parts) != 3:
                logging.error(f"Invalid FEN: {fen}, wrong number of parts")
                return False
            board, side, move_count = parts
            if side not in ['w', 'b']:
                logging.error(f"Invalid FEN: {fen}, invalid side {side}")
                return False
            if not move_count.isdigit():
                logging.error(f"Invalid FEN: {fen}, invalid move count {move_count}")
                return False
            rows = board.split('/')
            if len(rows) != 10:
                logging.error(f"Invalid FEN: {fen}, wrong number of rows")
                return False
            for row in rows:
                col_count = 0
                for char in row:
                    if char.isdigit():
                        col_count += int(char)
                    elif char.lower() in ['k', 'r', 'c', 'h', 'e', 'a', 'p']:
                        col_count += 1
                    else:
                        logging.error(f"Invalid FEN: {fen}, invalid character {char}")
                        return False
                if col_count != 9:
                    logging.error(f"Invalid FEN: {fen}, wrong column count in row")
                    return False
            return True
        except Exception as e:
            logging.error(f"FEN validation error: {e}")
            return False

    def convert_cdb_move_to_game_format(self, cdb_move, game):
        if not cdb_move or len(cdb_move) != 4:
            logging.warning(f"Invalid move format: {cdb_move}")
            return None
        try:
            src_col = ord(cdb_move[0].lower()) - ord('a')
            src_row = int(cdb_move[1])
            dst_col = ord(cdb_move[2].lower()) - ord('a')
            dst_row = int(cdb_move[3])
        except (ValueError, TypeError):
            logging.warning(f"Invalid coordinates in move: {cdb_move}")
            return None
        src_y = 9 - src_row
        dst_y = 9 - dst_row
        src_x = src_col
        dst_x = dst_col
        if not (0 <= src_x <= 8 and 0 <= dst_x <= 8 and 0 <= src_y <= 9 and 0 <= dst_y <= 9):
            logging.warning(f"Out of board coordinates in move: {cdb_move}")
            return None
        board = game.get_board()
        piece = board[src_y][src_x]
        if not piece or piece.is_red != game.is_red_move():
            logging.warning(f"No valid piece at position ({src_y}, {src_x}) for {'Red' if game.is_red_move() else 'Black'}: {cdb_move}")
            return None
        if piece.kind == 'h':
            dy, dx = dst_y - src_y, dst_x - src_x
            valid_horse_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            if (dy, dx) not in valid_horse_moves:
                logging.warning(f"Invalid horse move coordinates: {cdb_move}")
                return None
            if abs(dy) == 2:
                mid_y = src_y + (dy // 2)
                if board[mid_y][src_x] is not None:
                    logging.warning(f"Horse move blocked at ({mid_y}, {src_x}): {cdb_move}")
                    return None
            elif abs(dx) == 2:
                mid_x = src_x + (dx // 2)
                if board[src_y][mid_x] is not None:
                    logging.warning(f"Horse move blocked at ({src_y}, {mid_x}): {cdb_move}")
                    return None
        try:
            from Chinese_Chess_Game_Rules import _get_wxf_movement
            wxf_move = _get_wxf_movement(board, (src_y, src_x), (dst_y, dst_x), game.is_red_move())
        except Exception as e:
            logging.warning(f"Failed to convert move {cdb_move} to WXF: {str(e)}")
            return None
        valid_moves = game.get_valid_moves()
        if wxf_move not in valid_moves:
            logging.debug(f"Converted move {wxf_move} from {cdb_move} not in valid_moves")
            return None
        logging.debug(f"Successfully converted {cdb_move} to {wxf_move}")
        return wxf_move

    def fetch_moves_from_cdb(self, fen, max_retries=10):
        if fen in self.api_cache:
            moves = self.api_cache[fen]
            if moves:
                logging.info(f"Retrieved {len(moves)} moves from cache for FEN: {fen}")
                return moves
            else:
                logging.debug(f"Empty move list in cache for FEN: {fen}, retrying API")

        for attempt in range(max_retries):
            try:
                queryall_url = "http://www.chessdb.cn/chessdb.php"
                queryall_params = {
                    "action": "queryall",
                    "board": fen,
                    "showall": 0,
                    "learn": 1
                }
                logging.info(f"Attempt {attempt + 1}: Sending queryall request with FEN: {fen}")
                response = self.session.get(queryall_url, params=queryall_params, timeout=30)
                logging.info(f"Queryall status code: {response.status_code}")
                moves_with_scores = []
                if response.text in ["invalid board", "checkmate", "stalemate"]:
                    logging.warning(f"Queryall response: {response.text} for FEN: {fen}")
                    self.api_cache[fen] = []
                    self.api_fail_count += 1
                    self.save_api_cache()
                    return []
                elif response.text == "unknown":
                    logging.warning(f"Queryall response: unknown for FEN: {fen}")
                    if attempt < max_retries - 1:
                        logging.info("Retrying due to 'unknown' response")
                        time.sleep(2 ** attempt)
                        continue
                    self.api_cache[fen] = []
                    self.api_fail_count += 1
                    self.save_api_cache()
                    return []
                else:
                    move_entries = response.text.split("|")
                    for entry in move_entries:
                        if not entry:
                            continue
                        try:
                            fields = dict(field.split(":") for field in entry.split(","))
                            move = fields.get("move", "")
                            score = int(fields.get("score", "0"))
                            if move:
                                moves_with_scores.append((move, score))
                        except Exception as e:
                            logging.error(f"Error parsing entry {entry}: {e}")
                            continue
                    if not moves_with_scores:
                        logging.warning(f"No valid moves parsed from ChessDB response for FEN: {fen}")
                        self.api_cache[fen] = []
                        self.api_fail_count += 1
                        self.save_api_cache()
                        return []

                    # Điều chỉnh Score và tính xác suất bằng softmax
                    scores = np.array([score for _, score in moves_with_scores])
                    scores = np.clip(scores, -10, 10)  # Giới hạn Score
                    tau = 50  # Tăng tau để giảm độ chênh lệch
                    exp_scores = np.exp(scores / tau)
                    priors = exp_scores / np.sum(exp_scores)
                    moves = [move for move, _ in moves_with_scores]
                    moves_with_priors = list(zip(moves, priors))
                    logging.info(f"Retrieved {len(moves)} moves from ChessDB with priors: {[(move, float(prior)) for move, prior in moves_with_priors]}")
                    self.api_cache[fen] = moves_with_priors
                    self.api_success_count += 1
                    self.save_api_cache()
                    return moves_with_priors
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: Error fetching from CDB: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying after {2 ** attempt}s")
                    time.sleep(2 ** attempt)
                    continue
                self.api_cache[fen] = []
                self.api_fail_count += 1
                self.save_api_cache()
                logging.error(f"All {max_retries} attempts failed for FEN: {fen}")
                return []
        logging.error(f"All {max_retries} attempts failed for FEN: {fen}")
        return []
        
    def calculate_reward(self, game, prev_board, curr_board):
        reward = 0.0
        winner = game.get_winner()
        move_count = game._move_count
        decay_factor_early = np.exp(-0.001 * move_count)  # Decay chậm cho giai đoạn đầu
        decay_factor_late = np.exp(-0.005 * move_count)   # Decay nhanh cho giai đoạn cuối

        # Kiểm tra kết thúc ván cờ
        if winner:
            if winner == 'Red' and game.is_red_move() or winner == 'Black' and not game.is_red_move():
                reward = 10.0  # Phần thưởng lớn hơn cho chiến thắng
            elif winner == 'Draw':
                reward = 0.0
            else:
                reward = -10.0
            logging.info(f"Winner detected: {winner}, Reward: {reward}")
            return reward

        # Kiểm tra chiếu bí hoặc chiếu tướng
        opponent_in_check = game.is_in_check(curr_board, not game.is_red_move())
        if opponent_in_check:
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                checkmate_reward = 5.0 * decay_factor_early
                reward += checkmate_reward
                logging.debug(f"Checkmate achieved, Reward += {checkmate_reward}")
            else:
                num_escape_moves = len(valid_moves)
                check_reward = min(3.0, 2.0 / (num_escape_moves + 1)) * decay_factor_late
                reward += check_reward
                logging.debug(f"Check on opponent, Escape moves: {num_escape_moves}, Reward += {check_reward}")
        if game.is_in_check(curr_board, game.is_red_move()):
            check_penalty = 2.0 * decay_factor_early
            reward -= check_penalty
            logging.debug(f"Check on self, Reward -= {check_penalty}")

        # Tính giá trị quân cờ và kiểm soát không gian
        piece_types = ['k', 'r', 'c', 'h', 'e', 'a', 'p']
        prev_counts = {kind: {'red': 0, 'black': 0} for kind in piece_types}
        curr_counts = {kind: {'red': 0, 'black': 0} for kind in piece_types}
        prev_material = {'red': 0, 'black': 0}
        curr_material = {'red': 0, 'black': 0}
        prev_pawns_over_river = {'red': 0, 'black': 0}
        curr_pawns_over_river = {'red': 0, 'black': 0}
        #control_squares = {'red': 0, 'black': 0}
        critical_positions = []

        for r in range(10):
            for c in range(9):
                prev_piece = prev_board[r][c]
                curr_piece = curr_board[r][c]
                if prev_piece:
                    side = 'red' if prev_piece.is_red else 'black'
                    prev_counts[prev_piece.kind][side] += 1
                    prev_material[side] += abs(PIECE_TO_NUMBER[(prev_piece.kind, prev_piece.is_red)])
                    if prev_piece.kind == 'p' and ((side == 'red' and r <= 4) or (side == 'black' and r >= 5)):
                        prev_pawns_over_river[side] += 1
                if curr_piece:
                    side = 'red' if curr_piece.is_red else 'black'
                    curr_counts[curr_piece.kind][side] += 1
                    curr_material[side] += abs(PIECE_TO_NUMBER[(curr_piece.kind, curr_piece.is_red)])
                    if curr_piece.kind == 'p' and ((side == 'red' and r <= 4) or (side == 'black' and r >= 5)):
                        curr_pawns_over_river[side] += 1
                    if curr_piece.kind in ['r', 'c', 'h'] and curr_piece.is_red == game.is_red_move():
                        critical_positions.append((r, c))
                    # Phần thưởng vị trí chiến lược
                    if curr_piece.kind in ['r', 'c', 'h'] and 3 <= c <= 5 and 3 <= r <= 7:
                        position_reward = 0.7 * decay_factor_late if curr_piece.is_red == game.is_red_move() else -0.7 * decay_factor_late
                        reward += position_reward
                        logging.debug(f"Critical piece in center, Reward += {position_reward}")
                    if curr_piece.kind in ['a', 'e'] and 3 <= c <= 5 and (r <= 2 or r >= 7):
                        position_reward = 0.5 * decay_factor_late if curr_piece.is_red == game.is_red_move() else -0.5 * decay_factor_late
                        reward += position_reward
                        logging.debug(f"Advisor/elephant in center, Reward += {position_reward}")
                    if curr_piece.kind in ['r', 'c', 'h'] and ((r <= 2 and not curr_piece.is_red) or (r >= 7 and curr_piece.is_red)):
                        position_reward = 0.8 * decay_factor_late if curr_piece.is_red == game.is_red_move() else -0.8 * decay_factor_late
                        reward += position_reward
                        logging.debug(f"Critical piece in opponent half, Reward += {position_reward}")


        if game.is_red_move():
            new_pawns_over_river = curr_pawns_over_river['red'] - prev_pawns_over_river['red']
            if new_pawns_over_river > 0:
                pawn_reward = 0.4 * decay_factor_early * new_pawns_over_river
                reward += pawn_reward
                logging.debug(f"Red pawn crossed river, Reward += {pawn_reward}")
        else:
            new_pawns_over_river = curr_pawns_over_river['black'] - prev_pawns_over_river['black']
            if new_pawns_over_river > 0:
                pawn_reward = 0.4 * decay_factor_early * new_pawns_over_river
                reward += pawn_reward
                logging.debug(f"Black pawn crossed river, Reward += {pawn_reward}")

        # Phần thưởng và phạt cho ăn quân
        for kind in piece_types:
            value = abs(PIECE_TO_NUMBER[(kind, True)])
            if game.is_red_move():
                captured = prev_counts[kind]['black'] - curr_counts[kind]['black']
                if captured > 0:
                    capture_reward = min(3.0, np.log(1 + value)) * captured * decay_factor_early
                    reward += capture_reward
                    logging.debug(f"Red captured {captured} {kind}, Reward += {capture_reward}")
                lost = prev_counts[kind]['red'] - curr_counts[kind]['red']
                if lost > 0:
                    lost_penalty = np.log(1 + value) * lost * decay_factor_early
                    reward -= lost_penalty
                    logging.debug(f"Red lost {lost} {kind}, Reward -= {lost_penalty}")
            else:
                captured = prev_counts[kind]['red'] - curr_counts[kind]['red']
                if captured > 0:
                    capture_reward = min(3.0, np.log(1 + value)) * captured * decay_factor_early
                    reward += capture_reward
                    logging.debug(f"Black captured {captured} {kind}, Reward += {capture_reward}")
                lost = prev_counts[kind]['black'] - curr_counts[kind]['black']
                if lost > 0:
                    lost_penalty = np.log(1 + value) * lost * decay_factor_early
                    reward -= lost_penalty
                    logging.debug(f"Black lost {lost} {kind}, Reward -= {lost_penalty}")

        # Phần thưởng bảo vệ Tướng
        king_safe = 0
        for r in range(10):
            for c in range(9):
                piece = curr_board[r][c]
                if piece and piece.kind == 'k' and piece.is_red == game.is_red_move():
                    advisors = sum(1 for dr, dc in [(0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                                if 0 <= r + dr < 10 and 0 <= c + dc < 9 and
                                curr_board[r + dr][c + dc] and
                                curr_board[r + dr][c + dc].kind == 'a' and
                                curr_board[r + dr][c + dc].is_red == piece.is_red)
                    elephants = sum(1 for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]
                                    if 0 <= r + dr < 10 and 0 <= c + dc < 9 and
                                    curr_board[r + dr][c + dc] and
                                    curr_board[r + dr][c + dc].kind == 'e' and
                                    curr_board[r + dr][c + dc].is_red == piece.is_red)
                    king_safe = (advisors * 0.5 + elephants * 0.3) * decay_factor_late
                    reward += king_safe
                    logging.debug(f"King safety: Advisors={advisors}, Elephants={elephants}, Reward += {king_safe}")

        # Chuẩn hóa phần thưởng
        if abs(reward) > 10.0:
            logging.warning(f"Abnormal reward detected: {reward}, clamping to ±10.0")
            reward = np.sign(reward) * 10.0
        logging.debug(f"Final calculated reward: {reward}, Early decay: {decay_factor_early}, Late decay: {decay_factor_late}")
        return reward

    def save_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
        if len(self.replay_buffer) > 105000:
            self.replay_buffer.pop(0)
        logging.debug(f"Saved experience: Action={action}, Reward={reward}, Buffer size={len(self.replay_buffer)}")

    def create_critical_position(self):
        game = ChessGame()
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            logging.error("No valid moves for critical position")
            return None
        fen = self.board_to_fen(game)
        if not self.validate_fen(fen):
            logging.error(f"Invalid FEN in create_critical_position: {fen}")
            return None
        api_moves = self.fetch_moves_from_cdb(fen)
        converted_moves = [self.convert_cdb_move_to_game_format(move, game) for move in api_moves]
        valid_api_moves = [move for move in converted_moves if move]
        if not valid_api_moves:
            logging.warning("No valid API moves for critical position, using opening moves")
            valid_api_moves = [move for move in self.opening_moves if move in valid_moves]
            if not valid_api_moves:
                logging.warning("No valid opening moves, using policy network")
                state = self.board_to_tensor(game)
                with torch.no_grad():
                    probs = self.policy_network(state.unsqueeze(0)).cpu().numpy()[0]
                legal_probs = [probs[self.encode_move(move, valid_moves)] for move in valid_moves]
                legal_probs = np.array(legal_probs) / np.sum(legal_probs)
                valid_api_moves = valid_moves
        logging.info(f"Creating critical position with {len(valid_api_moves)} moves")
        for i in range(10):  # Giới hạn tối đa 10 nước đi
            if not valid_api_moves or game.get_winner():
                logging.warning("No valid moves or game ended, stopping simulation")
                break
            move = random.choice(valid_api_moves)
            prev_board = game.get_board()
            game = game.copy_and_make_move(move)
            reward = self.calculate_reward(game, prev_board, game.get_board())
            if abs(reward) >= 1.0 or game.is_in_check(game.get_board(), not game.is_red_move()):
                logging.info(f"Critical move: {move}, Reward: {reward}")
                break
            fen = self.board_to_fen(game)
            if not self.validate_fen(fen):
                logging.error(f"Invalid FEN after move {move}: {fen}")
                return None
            api_moves = self.fetch_moves_from_cdb(fen)
            converted_moves = [self.convert_cdb_move_to_game_format(move, game) for move in api_moves]
            valid_api_moves = [move for move in converted_moves if move]
            if not valid_api_moves:
                logging.warning("API failed in loop, using opening moves or random")
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break
                valid_api_moves = [move for move in self.opening_moves if move in valid_moves]
                if not valid_api_moves:
                    valid_api_moves = valid_moves
        state = self.board_to_tensor(game)
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            logging.error("No valid moves for final critical position")
            return None
        if valid_api_moves:
            move = random.choice(valid_api_moves)
        else:
            logging.warning("No API moves for final position, using policy network")
            with torch.no_grad():
                probs = self.policy_network(state.unsqueeze(0)).cpu().numpy()[0]
            legal_probs = [probs[self.encode_move(move, valid_moves)] for move in valid_moves]
            legal_probs = np.array(legal_probs) / np.sum(legal_probs)
            move = valid_moves[np.random.choice(len(valid_moves), p=legal_probs)]
        action_idx = self.encode_move(move, valid_moves)
        next_game = game.copy_and_make_move(move)
        next_state = self.board_to_tensor(next_game)
        reward = self.calculate_reward(next_game, game.get_board(), next_game.get_board())
        if next_game.is_in_check(next_game.get_board(), not next_game.is_red_move()):
            reward = 8.0
            logging.info(f"Critical position created with check, Reward: {reward}, Move: {move}")
        with torch.no_grad():
            value = self.value_network(state.unsqueeze(0)).cpu().numpy()[0].item()
            logging.info(f"Critical position: FEN={fen}, Move={move}, Reward={reward}, Value prediction={value}")
        return (state, action_idx, reward, next_state)

    def set_training_mode(self, mode=True):
        self.policy_network.train(mode)
        self.value_network.train(mode)
        logging.debug(f"Set network to {'training' if mode else 'evaluation'} mode")

    def train_main_network(self, batch_size=256):
        if len(self.replay_buffer) < 64:
            logging.info(f"Not enough samples: {len(self.replay_buffer)}/64")
            return
        batch_size = min(batch_size, len(self.replay_buffer))
        self_play = []
        len_self_play = int(batch_size * 0.2)  # Tăng tỷ lệ dữ liệu tổng hợp
        for _ in range(len_self_play):
            result = self.create_critical_position()
            if result:
                self_play.append(result)
                logging.info(f"Added critical position to batch, Reward: {result[2]}")
        index = random.sample(range(len(self.replay_buffer)), batch_size - len_self_play)
        exp_replay = [self.replay_buffer[i] for i in index]
        exp_replay += self_play
        state_batch, action_batch, reward_batch, next_state_batch = zip(*exp_replay)
        state_batch = torch.stack(state_batch).to(self.device).float()
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)
        next_state_batch = torch.stack(next_state_batch).to(self.device).float()

        logging.info(f"Training batch: Size={batch_size}, Buffer size={len(self.replay_buffer)}")
        logging.info(f"Reward stats: Mean={reward_batch.mean():.4f}, Std={reward_batch.std():.4f}, Min={reward_batch.min():.4f}, Max={reward_batch.max():.4f}")
        with torch.no_grad():
            policy_output = self.policy_network(state_batch)
            action_prob = policy_output.gather(1, action_batch.view(-1, 1)).squeeze()
            logging.info(f"Policy probs: Mean={action_prob.mean():.4f}, Std={action_prob.std():.4f}, Min={action_prob.min():.4f}, Max={action_prob.max():.4f}")
            if action_prob.min() < 1e-10 or action_prob.max() > 1.0:
                logging.warning("Abnormal policy probabilities detected")
            value_output = self.value_network(state_batch).squeeze()
            logging.info(f"Value predictions: Mean={value_output.mean():.4f}, Std={value_output.std():.4f}, Min={value_output.min():.4f}, Max={value_output.max():.4f}")

        self.set_training_mode(True)
        self.policy_optimizer.zero_grad()
        policy_output = self.policy_network(state_batch)
        action_prob = policy_output.gather(1, action_batch.view(-1, 1)).squeeze()
        action_prob = torch.clamp(action_prob, 1e-10, 1.0)
        policy_loss = -torch.mean(torch.log(action_prob) * (reward_batch.abs() + 1.0))  # Tăng trọng số cho phần thưởng lớn
        self.value_optimizer.zero_grad()
        value_output = self.value_network(state_batch).squeeze()
        value_loss = nn.MSELoss()(value_output, reward_batch)
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        logging.info(f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
        if self.epsilon * self.epsilon_decay > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            logging.debug(f"Epsilon updated: {self.epsilon:.4f}")
        self.update_frequency -= 1
        if self.update_frequency == 0:
            self.value_target_network.load_state_dict(self.value_network.state_dict())
            self.policy_target_network.load_state_dict(self.policy_network.state_dict())
            self.update_frequency = 10
            logging.info("Target networks updated")
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
        logging.info(f"Model saved at {model_path}")
        logging.info(f"API stats: Success={self.api_success_count}, Fail={self.api_fail_count}, Success rate={self.api_success_count / (self.api_success_count + self.api_fail_count + 1e-10):.4f}")

    def print_board(self, game):
        print_board(game.get_board())

    def get_move(self, game, simulations=5000, c_puct=2.0):  # Tăng simulations lên 5000
        if self.api_fail_count > self.api_fail_threshold:
            logging.warning(f"API failed {self.api_fail_count} times, switching to opening moves or policy network")
            valid_moves = game.get_valid_moves()
            if not valid_moves:
                logging.info("No valid moves available!")
                return None
            opening_moves = [move for move in self.opening_moves if move in valid_moves]
            if opening_moves:
                return random.choice(opening_moves)
            state = self.board_to_tensor(game)
            with torch.no_grad():
                probs = self.policy_network(state.unsqueeze(0)).cpu().numpy()[0]
            legal_probs = [probs[self.encode_move(move, valid_moves)] for move in valid_moves]
            legal_probs = np.array(legal_probs) / np.sum(legal_probs)
            return valid_moves[np.random.choice(len(valid_moves), p=legal_probs)]

        root = MCTSNode(game)
        valid_moves = game.get_valid_moves()
        logging.info(f"Valid moves generated: {len(valid_moves)} moves")
        if not valid_moves:
            logging.info("No valid moves available!")
            return None
        fen = self.board_to_fen(game)
        if not self.validate_fen(fen):
            logging.error(f"Invalid FEN generated: {fen}")
            return None
        api_moves_with_priors = self.fetch_moves_from_cdb(fen, max_retries=10)
        logging.info(f"Retrieved {len(api_moves_with_priors)} moves from ChessDB API")

        converted_moves_with_priors = []
        for move, prior in api_moves_with_priors:
            converted_move = self.convert_cdb_move_to_game_format(move, game)
            if converted_move:
                converted_moves_with_priors.append((converted_move, prior))
        if not converted_moves_with_priors:
            logging.warning(f"No valid moves from API after 10 retries, using opening moves or policy network")
            opening_moves = [move for move in self.opening_moves if move in valid_moves]
            if opening_moves:
                return random.choice(opening_moves)
            state = self.board_to_tensor(game)
            with torch.no_grad():
                probs = self.policy_network(state.unsqueeze(0)).cpu().numpy()[0]
            legal_probs = [probs[self.encode_move(move, valid_moves)] for move in valid_moves]
            legal_probs = np.array(legal_probs) / np.sum(legal_probs)
            return valid_moves[np.random.choice(len(valid_moves), p=legal_probs)]

        # Kết hợp xác suất từ API và PolicyNetwork
        state = self.board_to_tensor(game)
        with torch.no_grad():
            policy_probs = self.policy_network(state.unsqueeze(0)).cpu().numpy()[0]
        for i, (move, prior) in enumerate(converted_moves_with_priors):
            move_idx = self.encode_move(move, valid_moves)
            converted_moves_with_priors[i] = (move, 0.5 * prior + 0.5 * policy_probs[move_idx])
        moves, priors = zip(*converted_moves_with_priors)
        priors = np.array(priors) / np.sum(priors)
        logging.info(f"API moves and priors: {list(zip(moves, priors))}")
        root.expand(moves, priors)
        for sim in range(simulations):
            node = root
            sim_game = copy.deepcopy(game)
            action_path = []
            while node.children:
                move = node.select(c=c_puct)
                if move is None:
                    break
                action_path.append((node, move))
                node = node.children[move]
                sim_game = sim_game.copy_and_make_move(move)
            if not sim_game.get_winner():
                valid_moves = sim_game.get_valid_moves()
                if valid_moves:
                    state_tensor = self.board_to_tensor(sim_game)
                    with torch.no_grad():
                        probs = self.policy_target_network(state_tensor.unsqueeze(0)).cpu().numpy()[0]
                    legal_probs = [probs[self.encode_move(move, valid_moves)] for move in valid_moves]
                    legal_probs = np.array(legal_probs) / np.sum(legal_probs)
                    node.expand(valid_moves, legal_probs)
                    move = valid_moves[np.random.choice(len(valid_moves), p=legal_probs)]
                    action_path.append((node, move))
                    node = node.children[move]
                    sim_game = sim_game.copy_and_make_move(move)
                prev_board = game.get_board()
                curr_board = sim_game.get_board()
                value = self.calculate_reward(sim_game, prev_board, curr_board)
                with torch.no_grad():
                    value_pred = self.value_target_network(self.board_to_tensor(sim_game).unsqueeze(0)).cpu().numpy()[0].item()
                value = 0.5 * value + 0.5 * value_pred
                logging.debug(f"MCTS simulation: Move={move}, Reward={value}, Value prediction={value_pred}")
            else:
                winner = sim_game.get_winner()
                value = 1.0 if (winner == 'Red' and game.is_red_move()) or (winner == 'Black' and not game.is_red_move()) else -1.0
                if winner == 'Draw':
                    value = 0.0
                logging.debug(f"MCTS simulation ended with winner: {winner}, Value={value}")
            for parent_node, move in action_path:
                parent_node.update(move, value)
            if not action_path:
                root.visits += 1
        best_move = max(root.N, key=root.N.get) if root.N else random.choice(moves)
        api_move_list = list(moves)
        if best_move not in api_move_list:
            logging.warning(f"MCTS selected move {best_move} not in API moves, choosing random API move")
            best_move = random.choice(api_move_list)
        # Lưu nước đi vào move_history
        self.move_history.append((best_move, None))
        if len(self.move_history) > 10:
            self.move_history.pop(0)
        logging.info(f"MCTS move stats: {[(move, root.N[move], root.Q[move]) for move in root.N]}")
        logging.info(f"Selected move: {best_move}")
        return best_move

    def simulate(self, game):
        sim_game = copy.deepcopy(game)
        move_count = 0
        while not sim_game.get_winner():
            valid_moves = sim_game.get_valid_moves()
            if not valid_moves:
                break
            state_tensor = self.board_to_tensor(sim_game)
            with torch.no_grad():
                probs = self.policy_target_network(state_tensor.unsqueeze(0)).cpu().numpy()[0]
            legal_probs = [probs[self.encode_move(move, valid_moves)] for move in valid_moves]
            legal_probs = np.array(legal_probs) / np.sum(legal_probs)
            move = valid_moves[np.random.choice(len(valid_moves), p=legal_probs)]
            sim_game = sim_game.copy_and_make_move(move)
            move_count += 1
            if move_count > 100:
                logging.warning("Simulation exceeded 100 moves, forcing termination")
                break
        winner = sim_game.get_winner()
        state_tensor = self.board_to_tensor(sim_game)
        with torch.no_grad():
            value = self.value_target_network(state_tensor.unsqueeze(0)).cpu().numpy()[0].item()
        if winner:
            value = 1.0 if (winner == 'Red' and game.is_red_move()) or (winner == 'Black' and not game.is_red_move()) else -1.0
            logging.info(f"Simulation ended with winner: {winner}, Value: {value}")
        else:
            logging.info(f"Simulation ended without winner, Value: {value}")
        return value
    
    def reset_api_fail_count(self):
        self.api_fail_count = 0
        logging.info("Reset API fail count")

def train_game(num_games=1000, batch_size=1028, simulations=2000):
    agent = ChessAgent(state_size=(10, 9), action_size=4000)
    for game_idx in range(num_games):
        env = ChessGame()
        agent.reset_api_fail_count()
        logging.info(f"\nStarting game {game_idx + 1}:")
        agent.print_board(env)
        valid_moves = env.get_valid_moves()
        fen = agent.board_to_fen(env)
        logging.info(f"Initial FEN: {fen}")
        logging.info(f"Valid moves for {'Red' if env.is_red_move() else 'Black'}: {len(valid_moves)} moves")
        move_count = 0
        while not env.get_winner():
            move = agent.get_move(env, simulations=simulations)
            if move is None:
                logging.info(f"Game stopped at move {move_count + 1}: No valid moves available")
                break
            state = agent.board_to_tensor(env)
            with torch.no_grad():
                policy_probs = agent.policy_network(state.unsqueeze(0)).cpu().numpy()[0]
                move_prob = policy_probs[agent.encode_move(move, env.get_valid_moves())]
                value_pred = agent.value_network(state.unsqueeze(0)).cpu().numpy()[0].item()
            logging.info(f"Move {move_count + 1}: {move}, Policy prob: {move_prob:.4f}, Value prediction: {value_pred:.4f}")
            action_idx = agent.encode_move(move, env.get_valid_moves())
            prev_board = env.get_board()
            env.make_move(move)
            next_state = agent.board_to_tensor(env)
            reward = agent.calculate_reward(env, prev_board, env.get_board())
            logging.info(f"Reward for move {move}: {reward:.4f}")
            agent.save_experience(state, action_idx, reward, next_state)
            agent.print_board(env)
            move_count += 1
            winner = env.get_winner()
            if winner:
                logging.info(f"Game ended: {winner}")
                final_reward = 8.0 if (winner == 'Red' and env.is_red_move()) or (winner == 'Black' and not env.is_red_move()) else -8.0
                if winner == 'Draw':
                    final_reward = 0.0
                logging.info(f"Final reward: {final_reward}")
                agent.save_experience(next_state, action_idx, final_reward, next_state)
                break
            current_fen = agent.board_to_fen(env)
            if not agent.validate_fen(current_fen):
                logging.error(f"Invalid FEN after move {move_count}: {current_fen}")
                break
            logging.info(f"Current FEN after move {move_count}: {current_fen}")
        logging.info(f"Game {game_idx + 1} completed, Replay buffer size: {len(agent.replay_buffer)}")
        agent.train_main_network(batch_size=batch_size)
        agent.save()

def test():
    board = [
        [_Piece('r', False), _Piece('h', False), _Piece('e', False), _Piece('a', False), _Piece('k', False), _Piece('a', False), _Piece('e', False), _Piece('h', False), _Piece('r', False)],
        [None for _ in range(0, 9)],
        [None, _Piece('c', False), None, None, None, _Piece('c', False), None, None, None],
        [_Piece('p', False), None, _Piece('p', False), None, _Piece('p', False), None, _Piece('p', False), None, _Piece('p', False)],
        [None for _ in range(0, 9)],
        [None, None, _Piece('p', True), None, None, None, _Piece('p', True), None, None],
        [_Piece('p', True), None, None, None, _Piece('p', True), None, None, None, _Piece('p', True)],
        [None, _Piece('c', True), None, None, None, None, None, _Piece('c', True), None],
        [None for _ in range(0, 9)],
        [_Piece('r', True), _Piece('h', True), _Piece('e', True), _Piece('a', True), _Piece('k', True), _Piece('a', True), _Piece('e', True), _Piece('h', True), _Piece('r', True)],
    ]
    game = ChessGame(board=board, red_active=False)
    agent = ChessAgent()
    moves = game.get_valid_moves()
    fen = agent.board_to_fen(game)
    game.__str__()
    logging.info(f"Test board FEN: {fen}")
    logging.info(f"Test moves size: {len(moves)}")
    logging.info(f"Test moves: {moves}")
    api = agent.fetch_moves_from_cdb(fen)
    api_moves = [(agent.convert_cdb_move_to_game_format(move, game), 1.0 / len(api)) for move in api if agent.convert_cdb_move_to_game_format(move, game)]
    logging.info(f"Test API moves: {api_moves}")

if __name__ == "__main__":
    train_game(num_games=100000, batch_size=1028, simulations=2000)
    #test()