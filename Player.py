import random
import math
from Chinese_Chess_Game_Rules import ChessGame, PIECES, calculate_absolute_points, _Piece, _get_index_movement
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import json
import os
from datetime import datetime

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
    def __init__(self, iterations=1000, exploration_weight=1.0, simulate_fn=None):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.simulate_fn = simulate_fn if simulate_fn else self.default_evaluate_board
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

    def evaluate_with_dqn(self, game):
        if self.simulate_fn == self.default_evaluate_board:
            return self.default_evaluate_board(game)
        state_tensor = self.board_to_tensor(game)
        q_values = self.simulate_fn.main_network.predict(state_tensor[np.newaxis, ...], verbose=0)
        return np.max(q_values)

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
                value = self.evaluate_with_dqn(node.game)
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

class DQNAgent:
    def __init__(self, state_size=(10, 9, len(PIECES)), action_size=200):
        self.state_size = state_size
        self.action_size = action_size
        
        self.replay_buffer = deque(maxlen=100000)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.update_targetnn_rate = 100
        self.batch_size = 64

        self.main_network = self.get_nn()
        self.target_network = self.get_nn()

        self.target_network.set_weights(self.main_network.get_weights())
        self.steps_since_update = 0

    def _initialize_buffer(self):
        """Thêm các experience ngẫu nhiên ban đầu vào buffer"""
        for _ in range(1000):  # Số lượng experience khởi tạo
            # Tạo state ngẫu nhiên
            state = np.random.random(self.state_size).astype(np.float32)
            action = np.random.randint(self.action_size)
            reward = np.random.uniform(-1, 1)
            next_state = np.random.random(self.state_size).astype(np.float32)
            done = random.random() < 0.1  # 10% xác suất kết thúc
            
            self.replay_buffer.append((state, action, reward, next_state, done))

    def get_nn(self):
        """Build the neural network architecture."""
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.state_size),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(512, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='huber', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def encode_move(self, move, valid_moves):
        return valid_moves.index(move) if move in valid_moves else 0
    
    def decode_move(self, index, valid_moves):
        return valid_moves[index] if index < len(valid_moves) else None
    
    def save_experience(self, state, action, reward, next_state, terminal):
        self.replay_buffer.append((state, action, reward, next_state, terminal))

    def get_batch_from_buffer(self, batch_size):
        exp_batch = random.sample(self.replay_buffer, batch_size)
        state_batch  = np.array([batch[0] for batch in exp_batch])
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch])
        terminal_batch = [batch[4] for batch in exp_batch]
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch
    
    def tensor_to_board(self, tensor):
        """Chuyển tensor về dạng board game"""
        board = [[None for _ in range(9)] for _ in range(10)]
        piece_keys = list(PIECES.keys())
        
        for y in range(10):
            for x in range(9):
                piece_idx = np.argmax(tensor[y, x])
                if piece_idx > 0:
                    kind, is_red = piece_keys[piece_idx - 1]
                    board[y][x] = _Piece(kind, is_red)
        return board

    def train_main_network(self):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(self.batch_size)

        # Lấy Q value của state hiện tại
        q_values = self.main_network.predict(state_batch, verbose=0)

        # Lấy Max Q values của state S' (State chuyển từ S với action A)
        next_q_values = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q_values, axis=1)

        for i in range(self.batch_size):
            new_q_values = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma *max_next_q[i]
            q_values[i][action_batch[i]] = new_q_values

        self.main_network.fit(state_batch, q_values, batch_size=self.batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network periodically
        self.steps_since_update += 1
        if self.steps_since_update >= self.update_targetnn_rate:
            self.target_network.set_weights(self.main_network.get_weights())
            self.steps_since_update = 0
    

    def make_decision(self, state, valid_moves, training=True):
        if training and random.uniform(0,1) < self.epsilon:
            return random.choice(valid_moves)

        state = state[np.newaxis, ...]
        q_values = self.main_network.predict(state, verbose=0)[0]
        
        # Filter Q-values for valid moves only
        valid_indices = [self.encode_move(move, valid_moves) for move in valid_moves]
        valid_q = [q_values[i] for i in valid_indices]
        
        # Return move with highest Q-value
        best_idx = np.argmax(valid_q)
        return valid_moves[best_idx]
    
    def save_model(self, model_path, target_path=None):
        """Lưu trọng số của cả main network và target network"""
        self.main_network.save_weights(model_path)
        if target_path:
            self.target_network.save_weights(target_path)
    
    def load_model(self, model_path, target_path=None):
        """Tải trọng số đã lưu"""
        self.main_network.load_weights(model_path)
        if target_path:
            self.target_network.load_weights(target_path)
        else:
            # Nếu chỉ có 1 file, dùng chung cho cả target network
            self.target_network.set_weights(self.main_network.get_weights())
    
    def save_experience_buffer(self, buffer_path):
        """Lưu experience buffer để tiếp tục train sau"""
        np.save(buffer_path, np.array(self.replay_buffer, dtype=object))
    
    def load_experience_buffer(self, buffer_path):
        """Tải experience buffer đã lưu"""
        buffer = np.load(buffer_path, allow_pickle=True)
        self.replay_buffer = deque(buffer, maxlen=self.replay_buffer.maxlen)

    def save_full_model(self, base_path):
        """Lưu toàn bộ thông tin cần thiết để khôi phục agent"""
        # Lưu trọng số model
        self.main_network.save_weights(f"{base_path}_main.weights.h5")
        self.target_network.save_weights(f"{base_path}_target.weights.h5")
        
        # Lưu experience buffer
        np.save(f"{base_path}_exp.npy", np.array(self.replay_buffer, dtype=object))
        
        # Lưu hyperparameters
        params = {
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'update_targetnn_rate': self.update_targetnn_rate
        }
        with open(f"{base_path}_params.json", 'w') as f:
            json.dump(params, f)
    
    @classmethod
    def load_full_model(cls, base_path, training=False):
        """Tải toàn bộ model từ file đã lưu"""
        # Đọc hyperparameters
        with open(f"{base_path}_params.json", 'r') as f:
            params = json.load(f)
        
        # Khởi tạo agent
        agent = cls(state_size=(10, 9, len(PIECES)))
        agent.gamma = params['gamma']
        agent.epsilon = params['epsilon']
        agent.epsilon_min = params['epsilon_min']
        agent.epsilon_decay = params['epsilon_decay']
        agent.learning_rate = params['learning_rate']
        agent.batch_size = params['batch_size']
        agent.update_targetnn_rate = params['update_targetnn_rate']
        
        # Tải trọng số model
        agent.main_network.load_weights(f"{base_path}_main.weights.h5")
        agent.target_network.load_weights(f"{base_path}_target.weights.h5")
        
        # Tải experience buffer
        exp_buffer = np.load(f"{base_path}_exp.npy", allow_pickle=True)
        agent.replay_buffer = deque(exp_buffer, maxlen=agent.replay_buffer.maxlen)
        
        return agent

if __name__ == '__main__':    
    # Khởi tạo và train
    dqn_agent = DQNAgent()
    #dqn_agent = DQNAgent.load_full_model("trained_models/chinese_chess_dqn")
    mcts_player = MCTSPlayer(simulate_fn=dqn_agent)
    dqn_agent._initialize_buffer()

    # Train model (ví dụ đơn giản)
    for i in range(200):
        game = ChessGame()
        print("Vòng", i + 1)
        while game.get_winner() is None:
            move = mcts_player.make_move(game, None)
            game.make_move(move)
            game.__str__()
            dqn_agent.train_main_network()  # Train từ experience buffer
            print(dqn_agent.epsilon)
        if i % 10 == 0:
            dqn_agent.save_model('dqn_model.weights.h5')
            dqn_agent.save_experience_buffer('dqn_experience.npy')
            mcts_player.save_training_data('mcts_experience.npy')
            dqn_agent.save_full_model('trained_models/chinese_chess_dqn')
        print(game.get_winner())
    # Sau khi train xong, lưu lại
    dqn_agent.save_model('dqn_model.weights.h5')
    dqn_agent.save_experience_buffer('dqn_experience.npy')
    mcts_player.save_training_data('mcts_experience.npy')
    dqn_agent.save_full_model('trained_models/chinese_chess_dqn')

    # # Khi muốn sử dụng lại
    # new_dqn_agent = DQNAgent()
    # new_dqn_agent.load_model('dqn_model.weights.h5')
    # new_dqn_agent.load_experience_buffer('dqn_experience.npy')
    # loaded_agent = DQNAgent.load_full_model('trained_models/chinese_chess_dqn')

    # new_mcts_player = MCTSPlayer(simulate_fn=new_dqn_agent)
    # new_mcts_player.load_training_data('mcts_experience.npy')

    # game = ChessGame()
    # while game.get_winner() is None:
    #     move = mcts_player.make_move(game, None)
    #     game.make_move(move)
    #     print(game)  # Hiển thị bàn cờ