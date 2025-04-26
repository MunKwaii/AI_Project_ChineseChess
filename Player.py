import copy
import gc
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Giả định rằng Chinese_Chess_Game_Rules đã được định nghĩa
from Chinese_Chess_Game_Rules import ChessGame, _Piece, PIECES, print_board

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
    def __init__(self, action_size=200):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 10 * 9, 256)
        self.fc2 = nn.Linear(256, action_size)

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
        self.fc1 = nn.Linear(128 * 10 * 9, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

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

    def select(self, c=1.5):
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
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
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
                return False
            board, side, move_count = parts
            if side not in ['w', 'b']:
                return False
            if not move_count.isdigit():
                return False
            rows = board.split('/')
            if len(rows) != 10:
                return False
            for row in rows:
                col_count = 0
                for char in row:
                    if char.isdigit():
                        col_count += int(char)
                    elif char.lower() in ['k', 'r', 'c', 'h', 'e', 'a', 'p']:
                        col_count += 1
                    else:
                        return False
                if col_count != 9:
                    return False
            return True
        except Exception as e:
            logging.error(f"FEN validation error: {e}")
            return False

    def convert_cdb_move_to_game_format(self, cdb_move, game):
        if not cdb_move or len(cdb_move) != 4:
            logging.warning(f"Invalid move format: {cdb_move}")
            return None
        from_pos, to_pos = cdb_move[:2], cdb_move[2:]
        try:
            from_col = ord(from_pos[0]) - ord('a')
            from_row = int(from_pos[1])
            to_col = ord(to_pos[0]) - ord('a')
            to_row = int(to_pos[1])
        except (ValueError, TypeError):
            logging.warning(f"Invalid coordinates in move: {cdb_move}")
            return None
        if not (0 <= from_col <= 8 and 0 <= from_row <= 9 and 0 <= to_col <= 8 and 0 <= to_row <= 9):
            logging.warning(f"Out of board coordinates in move: {cdb_move}")
            return None
        from_col_game = from_col
        from_row_game = 9 - from_row
        to_col_game = to_col
        to_row_game = 9 - to_row
        board = game.get_board()
        piece = board[from_row_game][from_col_game]
        if not piece:
            logging.warning(f"No piece at position ({from_row_game}, {from_col_game}) for move: {cdb_move}")
            return None
        piece_kind = piece.kind.lower()
        is_red = piece.is_red
        from_col_wxf = 9 - from_col_game if is_red else from_col_game + 1
        to_col_wxf = 9 - to_col_game if is_red else to_col_game + 1
        from_str = f"{piece_kind}{from_col_wxf}"
        row_diff = from_row_game - to_row_game
        col_diff = to_col_game - from_col_game
        logging.debug(f"Converting {cdb_move}: from ({from_row_game}, {from_col_game}) to ({to_row_game}, {to_col_game}), piece: {piece_kind}, is_red: {is_red}")
        if piece_kind == 'h':  # Quân mã
            if (abs(row_diff), abs(col_diff)) not in [(2, 1), (1, 2)]:
                logging.warning(f"Invalid horse move coordinates: {cdb_move}")
                return None
            if abs(row_diff) == 2:
                mid_row = (from_row_game + to_row_game) // 2
                if board[mid_row][from_col_game]:
                    logging.warning(f"Horse move blocked at ({mid_row}, {from_col_game}): {cdb_move}")
                    return None
            elif abs(col_diff) == 2:
                mid_col = (from_col_game + to_col_game) // 2
                if board[from_row_game][mid_col]:
                    logging.warning(f"Horse move blocked at ({from_row_game}, {mid_col}): {cdb_move}")
                    return None
            move = f"{from_str}.{to_col_wxf}"
        elif to_row_game == from_row_game:  # Di chuyển ngang
            move = f"{from_str}.{to_col_wxf}"
        elif to_col_game == from_col_game:  # Di chuyển dọc
            steps = abs(row_diff)
            if (is_red and row_diff > 0) or (not is_red and row_diff < 0):
                move = f"{from_str}+{steps}"
            else:
                move = f"{from_str}-{steps}"
        else:  # Di chuyển chéo hoặc đặc biệt
            if piece_kind in ['e', 'a']:  # Tượng, sĩ
                move = f"{from_str}.{to_col_wxf}"
            elif piece_kind == 'p':  # Tốt
                if to_col_game == from_col_game:
                    steps = abs(row_diff)
                    if (is_red and row_diff > 0) or (not is_red and row_diff < 0):
                        move = f"{from_str}+{steps}"
                    else:
                        move = f"{from_str}-{steps}"
                else:
                    move = f"{from_str}.{to_col_wxf}"
            else:
                logging.warning(f"Invalid move for piece {piece_kind}: {cdb_move}")
                return None
        valid_moves = game.get_valid_moves()
        if move not in valid_moves:
            logging.debug(f"Converted move {move} from {cdb_move} not in valid_moves: {valid_moves}")
            return None
        logging.debug(f"Converted {cdb_move} to {move}")
        return move

    def fetch_moves_from_cdb(self, fen, max_retries=5):
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
                response = self.session.get(queryall_url, params=queryall_params, timeout=20)
                logging.info(f"Queryall status code: {response.status_code}, Response: {response.text}")
                response.raise_for_status()
                moves = []
                if response.text in ["invalid board", "unknown", "checkmate", "stalemate"]:
                    logging.warning(f"Queryall response: {response.text} for FEN: {fen}")
                    if response.text == "unknown" and attempt < max_retries - 1:
                        logging.info("Retrying due to 'unknown' response")
                        time.sleep(2)
                        continue
                else:
                    move_entries = response.text.split("|")
                    for entry in move_entries:
                        if not entry:
                            continue
                        try:
                            fields = dict(field.split(":") for field in entry.split(","))
                            move = fields.get("move", "")
                            if move:
                                moves.append(move)
                        except Exception as e:
                            logging.error(f"Error parsing entry {entry}: {e}")
                time.sleep(5)
                logging.info(f"Retrieved {len(moves)} moves from ChessDB")
                return moves
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: Error fetching from CDB: {e} for FEN: {fen}")
                if attempt < max_retries - 1:
                    logging.info("Retrying after error")
                    time.sleep(2)
                    continue
                return []
        logging.error(f"All {max_retries} attempts failed for FEN: {fen}")
        return []

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
                    curr_counts[piece.kind][side] += 1
                    # Phần thưởng cho vị trí trung tâm
                    if piece.kind in ['r', 'c', 'h'] and 3 <= c <= 5 and 3 <= r <= 7:
                        reward += 0.2 * decay_factor if piece.is_red == game.is_red_move() else -0.2 * decay_factor
                    # Phần thưởng bảo vệ vua
                    if piece.kind in ['a', 'e'] and 3 <= c <= 5 and (r <= 2 or r >= 7):
                        reward += 0.15 * decay_factor if piece.is_red == game.is_red_move() else -0.15 * decay_factor
        for kind in piece_types:
            value = abs(PIECE_TO_NUMBER[(kind, True)]) / 4.0
            if game.is_red_move():
                if curr_counts[kind]['black'] == 0:
                    reward += value * decay_factor
            else:
                if curr_counts[kind]['red'] == 0:
                    reward += value * decay_factor
        if game.is_in_check(curr_board, not game.is_red_move()):
            reward += 0.5 * decay_factor
        if abs(reward) > 1.0:  # Giới hạn reward để tránh giá trị cực đại
            reward = np.sign(reward) * 1.0
        logging.debug(f"Calculated reward: {reward}")
        return reward

    def save_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))
        if len(self.replay_buffer) > 105000:
            self.replay_buffer.pop(0)

    def create_opening_move(self):
        game = ChessGame()
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        fen = self.board_to_fen(game)
        api_moves = self.fetch_moves_from_cdb(fen)
        converted_moves = [self.convert_cdb_move_to_game_format(move, game) for move in api_moves]
        valid_api_moves = [move for move in converted_moves if move]
        if not valid_api_moves:
            logging.error("No valid API moves for opening, cannot proceed")
            return None
        move = random.choice(valid_api_moves)
        state = self.board_to_tensor(game)
        action_idx = self.encode_move(move, valid_moves)
        next_game = game.copy_and_make_move(move)
        next_state = self.board_to_tensor(next_game)
        reward = self.calculate_reward(next_game, game.get_board(), next_game.get_board())
        return (state, action_idx, reward, next_state)

    def set_training_mode(self, mode=True):
        self.policy_network.train(mode)
        self.value_network.train(mode)
        logging.debug(f"Set network to {'training' if mode else 'evaluation'} mode")

    def train_main_network(self, batch_size=1024):
        if len(self.replay_buffer) < batch_size:
            logging.info(f"Not enough samples: {len(self.replay_buffer)}/{batch_size}")
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
        logging.info(f"Model saved at {model_path}")

    def print_board(self, game):
        print_board(game.get_board())

    def get_move(self, game, simulations=1000, c_puct=1.5):
        root = MCTSNode(game)
        valid_moves = game.get_valid_moves()
        logging.info(f"Valid moves generated: {valid_moves}")
        if not valid_moves:
            logging.info("No valid moves available!")
            return None
        fen = self.board_to_fen(game)
        if not self.validate_fen(fen):
            logging.error(f"Invalid FEN generated: {fen}")
            return None
        logging.debug(f"Root FEN: {fen}")
        api_moves = self.fetch_moves_from_cdb(fen, max_retries=5)
        logging.info(f"Retrieved {len(api_moves)} moves from ChessDB API: {api_moves}")
        converted_moves = [(self.convert_cdb_move_to_game_format(move, game), 1.0 / len(api_moves)) for move in api_moves if self.convert_cdb_move_to_game_format(move, game)]
        logging.info(f"Converted {len(converted_moves)} moves: {converted_moves}")
        if not converted_moves:
            logging.warning(f"No valid moves from API after {5} retries, selecting random valid move")
            return random.choice(valid_moves) if valid_moves else None
        moves, priors = zip(*converted_moves)
        priors = np.array(priors) / np.sum(priors)
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
                sim_fen = self.board_to_fen(sim_game)
                logging.debug(f"MCTS move: {move}, Sim FEN: {sim_fen}")
            if not sim_game.get_winner():
                valid_moves = sim_game.get_valid_moves()
                if valid_moves:
                    sim_fen = self.board_to_fen(sim_game)
                    sim_api_moves = self.fetch_moves_from_cdb(sim_fen, max_retries=1)
                    sim_converted_moves = [(self.convert_cdb_move_to_game_format(move, sim_game), 1.0 / len(sim_api_moves)) if self.convert_cdb_move_to_game_format(move, sim_game) else None for move in sim_api_moves]
                    sim_converted_moves = [m for m in sim_converted_moves if m]
                    if sim_converted_moves:
                        moves, priors = zip(*sim_converted_moves)
                        priors = np.array(priors) / np.sum(priors)
                    else:
                        logging.warning("No valid API moves in simulation, skipping expansion")
                        continue
                    node.expand(moves, priors)
                    move = moves[np.random.choice(len(moves), p=priors)]
                    action_path.append((node, move))
                    node = node.children[move]
                    sim_game = sim_game.copy_and_make_move(move)
                    logging.debug(f"Expanded move: {move}, New Sim FEN: {self.board_to_fen(sim_game)}")
                prev_board = game.get_board()
                curr_board = sim_game.get_board()
                value = self.calculate_reward(sim_game, prev_board, curr_board)
            else:
                winner = sim_game.get_winner()
                value = 1.0 if (winner == 'Red' and game.is_red_move()) or (winner == 'Black' and not game.is_red_move()) else -1.0
                if winner == 'Draw':
                    value = 0.0
            for parent_node, move in action_path:
                parent_node.update(move, value)
            if not action_path:
                root.visits += 1
        best_move = max(root.N, key=root.N.get) if root.N else random.choice(moves)
        api_move_list = list(moves)
        if best_move not in api_move_list:
            logging.warning(f"MCTS selected move {best_move} not in API moves, choosing random API move")
            best_move = random.choice(api_move_list)
            logging.info(f"Selected random API move: {best_move}")
        for move in root.N:
            logging.info(f"Move: {move}, Visits: {root.N[move]}, Q-value: {root.Q[move]:.4f}")
        logging.info(f"Selected move: {best_move}")
        return best_move

    def simulate(self, game):
        sim_game = copy.deepcopy(game)
        while not sim_game.get_winner():
            valid_moves = sim_game.get_valid_moves()
            if not valid_moves:
                break
            fen = self.board_to_fen(sim_game)
            api_moves = self.fetch_moves_from_cdb(fen, max_retries=1)
            converted_moves = [self.convert_cdb_move_to_game_format(move, sim_game) for move in api_moves]
            valid_api_moves = [move for move in converted_moves if move]
            if not valid_api_moves:
                logging.warning("No valid API moves in simulation, selecting random valid move")
                move = random.choice(valid_moves)
            else:
                move = random.choice(valid_api_moves)
            sim_game = sim_game.copy_and_make_move(move)
        winner = sim_game.get_winner()
        state_tensor = self.board_to_tensor(sim_game)
        with torch.no_grad():
            value = self.value_target_network(state_tensor.unsqueeze(0)).cpu().numpy()[0]
        if winner:
            return 1.0 if (winner == 'Red' and game.is_red_move()) or (winner == 'Black' and not game.is_red_move()) else -1.0
        return value

# Hàm huấn luyện chính
def train_game(num_games=1, batch_size=1024, simulations=1000):
    agent = ChessAgent(state_size=(10, 9), action_size=200)
    for game_idx in range(num_games):
        env = ChessGame()
        print(f"\nStarting game {game_idx + 1}:")
        agent.print_board(env)
        valid_moves = env.get_valid_moves()
        print(f"Valid moves for {'Red' if env.is_red_move() else 'Black'}: {valid_moves}")
        print(f"Number of valid moves: {len(valid_moves)}")
        fen = agent.board_to_fen(env)
        print(f"Initial FEN: {fen}")
        move_count = 0
        while not env.get_winner():
            fen = agent.board_to_fen(env)
            api_moves = agent.fetch_moves_from_cdb(fen, max_retries=1)
            converted_moves = [agent.convert_cdb_move_to_game_format(move, env) for move in api_moves]
            valid_api_moves = [move for move in converted_moves if move]
            move = agent.get_move(env, simulations=simulations)
            if move is None:
                print(f"Game stopped at move {move_count + 1}: No valid moves available")
                break
            if move not in valid_api_moves:
                logging.warning(f"Move {move} not in API moves: {valid_api_moves}, but proceeding as fallback")
            print(f"Move {move_count + 1}: {move}")
            state = agent.board_to_tensor(env)
            action_idx = agent.encode_move(move, env.get_valid_moves())
            prev_board = env.get_board()
            env.make_move(move)
            next_state = agent.board_to_tensor(env)
            reward = agent.calculate_reward(env, prev_board, env.get_board())
            agent.save_experience(state, action_idx, reward, next_state)
            agent.print_board(env)
            move_count += 1
            winner = env.get_winner()
            if winner:
                print(f"Game ended: {winner}")
                final_reward = 8.0 if (winner == 'Red' and env.is_red_move()) or (winner == 'Black' and not env.is_red_move()) else -8.0
                if winner == 'Draw':
                    final_reward = 0.0
                agent.save_experience(next_state, action_idx, final_reward, next_state)
                break
            current_fen = agent.board_to_fen(env)
            logging.info(f"Current FEN after move {move_count}: {current_fen}")
        print(f"Training network after game {game_idx + 1}...")
        agent.train_main_network(batch_size=batch_size)
        agent.save()

if __name__ == "__main__":
    train_game(num_games=1, batch_size=1024, simulations=1000)