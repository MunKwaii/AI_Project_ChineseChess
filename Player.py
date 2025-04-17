import random
import math
import os
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import copy
import logging

# Giả sử Chinese_Chess_Game_Rules.py đã được cung cấp
from Chinese_Chess_Game_Rules import ChessGame, PIECES, _Piece, _get_index_movement

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dqn_training.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Lớp MCTSNode
class MCTSNode:
    def __init__(self, game, move=None, parent=None):
        self.game = game
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = 1.0
        self.lock = threading.Lock()

    def ucb1(self, exploration_weight=1.5):
        with self.lock:
            if self.visits == 0:
                return float('inf')
            return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits + 1) / (self.visits + 1))

    def expand(self):
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            logging.debug("No valid moves to expand")
            return None
        for move in valid_moves:
            new_game = self.game.copy_and_make_move(move)
            new_game.board_history = {}
            self.children.append(MCTSNode(new_game, move, self))
        logging.debug(f"Expanded node with {len(self.children)} children")
        return random.choice(self.children) if self.children else None

    def backpropagate(self, value):
        with self.lock:
            self.visits += 1
            self.value += value
        if self.parent:
            self.parent.backpropagate(-value)

# Lớp MCTSPlayer
class MCTSPlayer:
    def __init__(self, dqn_agent, iterations=1000, exploration_weight=1.5, max_workers=6):
        self.dqn_agent = dqn_agent
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.move_history = deque(maxlen=50)
        self.opening_move_index = {'Red': 0, 'Black': 0}
        self.opening_moves = {
            'Red': ["c2.5", "p3+1", "h2+3", "r1.2", "h8+9", "r2+6"],
            'Black': ["h8+7", "r9+1", "r9.4", "c8-1", "c8.5", "r4+6"]
        }
        self.max_workers = max_workers

    def reset_opening(self):
        self.opening_move_index = {'Red': 0, 'Black': 0}
        self.move_history.clear()

    def board_to_tensor(self, game):
        board = game.get_board()
        tensor = torch.zeros((10, 9, len(PIECES)), dtype=torch.float32)
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece:
                    key = (piece.kind, piece.is_red)
                    if key in PIECES:
                        tensor[r, c, list(PIECES.keys()).index(key)] = 1
        return tensor

    def random_rollout(self, game, max_depth=15):
        current_game = copy.deepcopy(game)
        current_game.board_history = {}
        depth = 0
        while depth < max_depth and current_game.get_valid_moves():
            valid_moves = current_game.get_valid_moves()
            if not valid_moves:
                logging.debug("No valid moves in rollout")
                break
            move = random.choice(valid_moves)
            current_game = current_game.copy_and_make_move(move)
            current_game.board_history = {}
            depth += 1
        winner = current_game.get_winner()
        if winner is not None:
            if winner == 'Red' and current_game.is_red_move() or winner == 'Black' and not current_game.is_red_move():
                return 1.0
            elif winner == 'Draw':
                return 0.0
            else:
                return -1.0
        return 0.0

    def simulate_node(self, root):
        node = root
        depth = 0
        max_depth = 50
        path = [node.move] if node.move else ["root"]
        logging.debug(f"Starting simulation from node with move: {node.move}")

        while node.children and depth < max_depth and node.game.get_valid_moves():
            node = max(node.children, key=lambda c: c.ucb1(self.exploration_weight))
            path.append(node.move)
            depth += 1
            logging.debug(f"Selected child node at depth {depth}, move: {node.move}, UCB1: {node.ucb1(self.exploration_weight):.4f}")

        if depth < max_depth and node.game.get_valid_moves():
            new_node = node.expand()
            if new_node:
                node = new_node
                path.append(node.move)
                depth += 1
            else:
                logging.debug("No expansion possible")
                value = 0.0
                node.backpropagate(value)
                return

        value = self.random_rollout(node.game, max_depth=max_depth - depth)
        logging.debug(f"Rollout completed, value: {value:.4f}")

        if node.game.get_valid_moves():
            state_tensor = self.board_to_tensor(node.game)
            state_tensor = state_tensor.unsqueeze(0).to(self.dqn_agent.device)
            state_tensor = state_tensor.permute(0, 3, 1, 2)
            with torch.no_grad():
                dqn_value = self.dqn_agent.main_network(state_tensor).max().item()
            value = dqn_value if dqn_value > 0.1 else value
            logging.debug(f"Evaluated node with DQN value: {dqn_value:.4f}, final value: {value:.4f}")

        node.backpropagate(value)

    def make_move(self, game, previous_move):
        start_time = time.perf_counter()
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            elapsed_time = time.perf_counter() - start_time
            logging.info(f"Move time: {elapsed_time:.4f} seconds (No valid moves)")
            return None

        #logging.info("Current board state:")
        game.__str__()

        if random.uniform(0, 1) < self.dqn_agent.epsilon:
            best_move = random.choice(valid_moves)
            logging.info(f"Epsilon-greedy: Random move selected: {best_move}")
        else:
            root = MCTSNode(copy.deepcopy(game))
            root.game.board_history = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.simulate_node, root) for _ in range(self.iterations)]
                for future in futures:
                    try:
                        future.result(timeout=20)
                    except TimeoutError:
                        logging.warning("Simulation thread timed out")

            if not root.children:
                elapsed_time = time.perf_counter() - start_time
                logging.info(f"Move time: {elapsed_time:.4f} seconds (Random choice)")
                return random.choice(valid_moves) if valid_moves else None

            best_move = max(root.children, key=lambda c: c.visits).move
            logging.info(f"MCTS selected move: {best_move}")

        self.move_history.append(best_move)
        elapsed_time = time.perf_counter() - start_time
        logging.info(f"Move time: {elapsed_time:.4f} seconds for move {best_move}")
        return best_move

# Lớp DQNAgent
class DQNAgent:
    def __init__(self, state_size=(10, 9, len(PIECES)), action_size=200):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10
        self.batch_size = 64 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device in DQNAgent: {self.device}")
        self.main_network = self.get_nn().to(self.device)
        self.target_network = self.get_nn().to(self.device)
        self.target_network.eval()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.total_time_step = 0
        self.episode = 0  # Khởi tạo episode

    def get_nn(self):
        class DQN(nn.Module):
            def __init__(self, state_size, action_size):
                super(DQN, self).__init__()
                self.conv1 = nn.Conv2d(state_size[2], 32, kernel_size=5, padding=2)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
                conv_out_size = state_size[0] * state_size[1] * 64
                self.fc1 = nn.Linear(conv_out_size, 256)
                self.fc2 = nn.Linear(256, action_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = x.reshape(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        return DQN(self.state_size, self.action_size)

    def board_to_tensor(self, game):
        board = game.get_board()
        tensor = torch.zeros((10, 9, len(PIECES)), dtype=torch.float32)
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece:
                    key = (piece.kind, piece.is_red)
                    if key in PIECES:
                        tensor[r, c, list(PIECES.keys()).index(key)] = 1
        return tensor

    def encode_move(self, move, valid_moves):
        return valid_moves.index(move) if move in valid_moves else 0

    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):
        exp_batch = random.sample(self.replay_buffer, batch_size)
        state_batch = torch.stack([batch[0] for batch in exp_batch])
        action_batch = torch.tensor([batch[1] for batch in exp_batch], dtype=torch.long)
        reward_batch = torch.tensor([batch[2] for batch in exp_batch], dtype=torch.float32)
        next_state_batch = torch.stack([batch[3] for batch in exp_batch])
        terminal_batch = torch.tensor([batch[4] for batch in exp_batch], dtype=torch.bool)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def train_main_network(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            logging.info(f"Not enough samples in replay buffer: {len(self.replay_buffer)}/{batch_size}")
            return
        logging.debug("Starting training of main network")
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)

        self.main_network.train()
        if dqn_agent.epsilon > dqn_agent.epsilon_min:dqn_agent.epsilon *= dqn_agent.epsilon_decay
        self.update_targetnn_rate-=1 #le
        if self.update_targetnn_rate==0:
            self.target_network.load_state_dict(self.main_network.state_dict())
            self.update_targetnn_rate==10
        state_batch = state_batch.to(self.device)
        state_batch = state_batch.permute(0, 3, 1, 2)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        next_state_batch = next_state_batch.permute(0, 3, 1, 2)
        terminal_batch = terminal_batch.to(self.device)

        def predict_batch(states):
            with torch.no_grad():
                return self.main_network(states)

        def predict_target_batch(states):
            with torch.no_grad():
                return self.target_network(states)

        batch_split = 4
        split_size = batch_size // batch_split
        q_values = torch.zeros((batch_size, self.action_size), device=self.device)
        next_q_values = torch.zeros((batch_size, self.action_size), device=self.device)

        with ThreadPoolExecutor(max_workers=batch_split) as executor:
            futures = []
            for i in range(0, batch_size, split_size):
                state_slice = state_batch[i:i+split_size]
                next_state_slice = next_state_batch[i:i+split_size]
                futures.append(executor.submit(predict_target_batch, state_slice))
                futures.append(executor.submit(predict_target_batch, next_state_slice))
            
            idx = 0
            for future in futures:
                result = future.result()
                if idx < batch_size:
                    q_values[idx:idx+split_size] = result
                else:
                    next_q_values[idx-batch_size:idx-batch_size+split_size] = result
                idx += split_size
                if idx >= batch_size:
                    idx = 0

        max_next_q = next_q_values.max(dim=1)[0]
        target_q = q_values.clone()
        for i in range(batch_size):
            new_q_values = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]
            target_q[i][action_batch[i]] = new_q_values

        self.optimizer.zero_grad()
        output = self.main_network(state_batch)
        loss = self.loss_fn(output, target_q)
        loss.backward()
        self.optimizer.step()

        logging.debug("Finished training batch")

    def save(self, base_path):
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        torch.save({
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "main_network": self.main_network.state_dict(),
        }, base_path)
        logging.info(f"Saved model at {base_path}")

    def load(self, base_path):
        if os.path.exists(base_path):
            open_model = torch.load(base_path, map_location=self.device)
            if isinstance(open_model, dict):
                logging.info("Successfully loaded checkpoint data")
                self.gamma = open_model.get("gamma")
                self.epsilon = open_model.get("epsilon")
                self.epsilon_min = open_model.get("epsilon_min")
                self.epsilon_decay = open_model.get("epsilon_decay")
                self.learning_rate = open_model.get("learning_rate")
                self.main_network.load_state_dict(open_model["main_network"])
                print(f"gia tri cua epsilon {self.epsilon}")
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()
# Chương trình chính
if __name__ == "__main__":
    env = ChessGame()
    env.board_history = {}
    state_size = (10, 9, len(PIECES))
    action_size = 200
    n_timesteps = 500
    batch_size = 64

    dqn_agent = DQNAgent(state_size=state_size, action_size=200)
    
    # Tải mô hình nếu có
    base_dir = "trained_models"
    model_path = os.path.join(base_dir, "chinese_chess_dqn.pth")
    dqn_agent.load(model_path)
    start_ep = dqn_agent.episode

    mcts_player = MCTSPlayer(dqn_agent=dqn_agent, iterations=1000, max_workers=15)
    
    ep = start_ep
    try:
        while True:
            ep += 1
            ep_rewards = 0
            env = ChessGame()
            env.board_history = {}
            mcts_player.reset_opening()
            state = dqn_agent.board_to_tensor(env)
            logging.info(f"\n=== Starting Episode {ep} ===")
            for t in range(n_timesteps):
                dqn_agent.total_time_step += 1
                action = mcts_player.make_move(env, env.last_move)
                if action is None:
                    logging.info("No valid moves available!")
                    break
                
                env.make_move(action)
                logging.info("Board after move:")
                env.__str__()
                next_state = dqn_agent.board_to_tensor(env)
                reward = 0.0  # Khởi tạo reward, không dùng default_evaluate_board, chỉ phạt mất quân
                board = env.get_board()
                red_cannons = sum(1 for y in range(10) for x in range(9) if board[y][x] and board[y][x].kind == 'cannon' and board[y][x].is_red)
                red_rooks = sum(1 for y in range(10) for x in range(9) if board[y][x] and board[y][x].kind == 'rook' and board[y][x].is_red)
                black_cannons = sum(1 for y in range(10) for x in range(9) if board[y][x] and board[y][x].kind == 'cannon' and not board[y][x].is_red)
                black_rooks = sum(1 for y in range(10) for x in range(9) if board[y][x] and board[y][x].kind == 'rook' and not board[y][x].is_red)
                if env.is_red_move():
                    if black_cannons < 2:
                        reward -= 0.03
                    if black_rooks < 2:
                        reward -= 0.05
                else:
                    if red_cannons < 2:
                        reward -= 0.03
                    if red_rooks < 2:
                        reward -= 0.05
                terminal = env.get_winner() is not None

                action_idx = dqn_agent.encode_move(action, env.get_valid_moves())
                dqn_agent.save_experience(state, action_idx, reward, next_state, terminal)

                state = next_state
                ep_rewards += reward

                logging.info(f"Move: {action} | Reward: {reward:.4f}")

                if terminal:
                    winner = env.get_winner()
                    logging.info(f"Episode {ep} ended with winner: {winner}, Total reward: {ep_rewards:.4f}")
                    break

                if len(dqn_agent.replay_buffer) > batch_size:
                    dqn_agent.train_main_network(batch_size)

            if not terminal:
                logging.info(f"Episode {ep} reached {t + 1} moves, Total reward: {ep_rewards:.4f}")

            # Lưu mô hình mỗi 10 episode
            if ep % 1 == 0:
                dqn_agent.episode = ep
                dqn_agent.save(model_path)
                logging.info(f"Saved model at episode {ep}")

    except KeyboardInterrupt:
        logging.info(f"Training interrupted at episode {ep}. Saving model...")
        dqn_agent.episode = ep
        dqn_agent.save(model_path)
        logging.info(f"Saved model at episode {ep}")
        raise