import random
import math
from Chinese_Chess_Game_Rules import ChessGame, PIECES, calculate_absolute_points, _Piece, _get_index_movement, piece_count
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import time  # Thêm import time để đo thời gian

logging.basicConfig(
    level=logging.DEBUG,
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

    def ucb1(self, exploration_weight=1.0):
        with self.lock:
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
        with self.lock:
            self.visits += 1
            self.value += value
        if self.parent:
            self.parent.backpropagate(-value)

# Lớp MCTSPlayer
class MCTSPlayer:
    def __init__(self, dqn_agent, iterations=1000, exploration_weight=2.0, max_workers=15):
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
        is_red = game.is_red_move()
        side_modifier = 1 if is_red else -1

        total_score = calculate_absolute_points(board)
        if game.is_in_check(board, is_red):
            total_score += (-200) * side_modifier
        if game.is_in_check(board, not is_red):
            total_score += 300 * side_modifier
        current_move = self.move_history[-1] if self.move_history else None
        if current_move and list(self.move_history).count(current_move) >= 2:
            total_score += (-300) * side_modifier
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece and piece.is_red == is_red and piece.kind in {'r', 'c'}:
                    if piece.is_red and y <= 2:
                        total_score += 100 * side_modifier
                    elif not piece.is_red and y >= 7:
                        total_score += 100 * side_modifier
        return min(max(total_score / 10000, -1), 1)

    def simulate_node(self, root):
        node = root
        while node.children and node.game.get_winner() is None:
            node = max(node.children, key=lambda c: c.ucb1(self.exploration_weight))
        if node.game.get_winner() is None:
            node = node.expand()
        if node:
            state_tensor = self.board_to_tensor(node.game)
            value = self.dqn_agent.main_network.predict(state_tensor[np.newaxis, ...], verbose=0).max()
            if value < 0.1:
                value = self.default_evaluate_board(node.game)
            node.backpropagate(value)

    def make_move(self, game, previous_move):
        start_time = time.perf_counter()  # Bắt đầu đo thời gian
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            elapsed_time = time.perf_counter() - start_time
            logging.info(f"Move time: {elapsed_time:.4f} seconds (No valid moves)")
            return None

        root = MCTSNode(game)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.simulate_node, root) for _ in range(self.iterations)]
            for future in futures:
                future.result()

        if not root.children:
            elapsed_time = time.perf_counter() - start_time
            logging.info(f"Move time: {elapsed_time:.4f} seconds (Random choice)")
            return random.choice(valid_moves) if valid_moves else None

        capturing_moves = []
        board = game.get_board()
        for child in root.children:
            y, x = _get_index_movement(board, child.move, game.is_red_move())
            captured = board[y][x]
            if captured and captured.is_red != game.is_red_move():
                capturing_moves.append((child.move, child.visits))

        best_move = max(capturing_moves, key=lambda x: x[1])[0] if capturing_moves else \
                    max(root.children, key=lambda c: c.visits).move
        self.move_history.append(best_move)
        
        elapsed_time = time.perf_counter() - start_time  # Tính thời gian thực hiện
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
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_targetnn_rate = 10
        self.batch_size = 64
        self.main_network = self.get_nn()
        self.target_network = self.get_nn()
        self.target_network.set_weights(self.main_network.get_weights())
        self.total_time_step = 0

    def get_nn(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.state_size),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

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

    def encode_move(self, move, valid_moves):
        return valid_moves.index(move) if move in valid_moves else 0

    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):
        exp_batch = random.sample(self.replay_buffer, batch_size)
        state_batch = np.array([batch[0] for batch in exp_batch])
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch])
        terminal_batch = [batch[4] for batch in exp_batch]
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def train_main_network(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            logging.info(f"Not enough samples in replay buffer: {len(self.replay_buffer)}/{batch_size}")
            return
        logging.debug("Starting training of main network")
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(batch_size)

        def predict_batch(states):
            return self.main_network.predict(states, verbose=0)

        def predict_target_batch(states):
            return self.target_network.predict(states, verbose=0)

        batch_split = 4
        split_size = batch_size // batch_split
        q_values = np.zeros((batch_size, self.action_size))
        next_q_values = np.zeros((batch_size, self.action_size))

        with ThreadPoolExecutor(max_workers=batch_split) as executor:
            futures = []
            for i in range(0, batch_size, split_size):
                state_slice = state_batch[i:i+split_size]
                next_state_slice = next_state_batch[i:i+split_size]
                futures.append(executor.submit(predict_batch, state_slice))
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

        max_next_q = np.amax(next_q_values, axis=1)
        for i in range(batch_size):
            new_q_values = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]
            q_values[i][action_batch[i]] = new_q_values

        self.main_network.fit(state_batch, q_values, verbose=0)
        logging.debug("Finished training batch")

    def make_decision(self, state, valid_moves):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        state = state[np.newaxis, ...]
        q_values = self.main_network.predict(state, verbose=0)[0]
        valid_indices = [self.encode_move(move, valid_moves) for move in valid_moves]
        valid_q = [q_values[i] for i in valid_indices]
        return valid_moves[np.argmax(valid_q)]

    def save(self, base_path):
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        self.main_network.save_weights(f"{base_path}_main.weights.h5")
        self.target_network.save_weights(f"{base_path}_target.weights.h5")
        np.save(f"{base_path}_exp.npy", np.array(self.replay_buffer, dtype=object))
        params = {
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'update_targetnn_rate': self.update_targetnn_rate,
            'total_time_step': self.total_time_step
        }
        with open(f"{base_path}_params.json", 'w') as f:
            json.dump(params, f)
        print(f"Đã lưu toàn bộ mô hình tại {base_path}")

    def load(self, base_path):
        self.main_network.load_weights(f"{base_path}_main.weights.h5")
        self.target_network.load_weights(f"{base_path}_target.weights.h5")
        buffer = np.load(f"{base_path}_exp.npy", allow_pickle=True)
        self.replay_buffer = deque(buffer, maxlen=self.replay_buffer.maxlen)
        with open(f"{base_path}_params.json", 'r') as f:
            params = json.load(f)
            self.gamma = params['gamma']
            self.epsilon = params['epsilon']
            self.epsilon_min = params['epsilon_min']
            self.epsilon_decay = params['epsilon_decay']
            self.learning_rate = params['learning_rate']
            self.batch_size = params['batch_size']
            self.update_targetnn_rate = params['update_targetnn_rate']
            self.total_time_step = params['total_time_step']
        print(f"Đã tải mô hình từ {base_path}")

# Chương trình chính
if __name__ == "__main__":
    env = ChessGame()
    state_size = (10, 9, len(PIECES))
    action_size = 200
    n_episodes = 100
    n_timesteps = 500
    batch_size = 64

    dqn_agent = DQNAgent(state_size=state_size, action_size=200)
    try:
        dqn_agent.load("trained_models/chinese_chess_dqn")
        logging.info("Loaded pre-trained model successfully")
    except Exception as e:
        logging.warning(f"Failed to load model: {e}. Using untrained model.")

    mcts_player = MCTSPlayer(dqn_agent=dqn_agent, iterations=500, max_workers=4)

    for ep in range(30, n_episodes):
        ep_rewards = 0
        env = ChessGame()
        mcts_player.reset_opening()
        state = dqn_agent.board_to_tensor(env)
        logging.info(f"\n=== Starting Episode {ep + 1} ===")

        for t in range(n_timesteps):
            dqn_agent.total_time_step += 1
            if dqn_agent.total_time_step % dqn_agent.update_targetnn_rate == 0:
                dqn_agent.target_network.set_weights(dqn_agent.main_network.get_weights())
                logging.debug("Updated target network weights")

            action = mcts_player.make_move(env, env.last_move)
            if action is None:
                logging.info("No valid moves available!")
                break

            env.make_move(action)
            env.__str__()
            next_state = dqn_agent.board_to_tensor(env)
            reward = mcts_player.default_evaluate_board(env)
            terminal = env.get_winner() is not None

            action_idx = dqn_agent.encode_move(action, env.get_valid_moves())
            dqn_agent.save_experience(state, action_idx, reward, next_state, terminal)

            state = next_state
            ep_rewards += reward

            logging.info(f"Move: {action} | Reward: {reward:.4f}")

            if terminal:
                winner = env.get_winner()
                logging.info(f"Episode {ep + 1} ended with winner: {winner}, Total reward: {ep_rewards:.4f}")
                break

            if len(dqn_agent.replay_buffer) > batch_size:
                dqn_agent.train_main_network(batch_size)

        if dqn_agent.epsilon > dqn_agent.epsilon_min:
            dqn_agent.epsilon *= dqn_agent.epsilon_decay
            logging.debug(f"Updated epsilon: {dqn_agent.epsilon:.4f}")

        if not terminal:
            logging.info(f"Episode {ep + 1} reached {t + 1} moves, Total reward: {ep_rewards:.4f}")

        if (ep + 1) % 10 == 0:
            dqn_agent.save(f"trained_models/chinese_chess_dqn")
            logging.info(f"Saved model at episode {ep + 1}")

    dqn_agent.main_network.save("trained_models/chinese_chess_dqn_final")
    logging.info("Saved final model.")