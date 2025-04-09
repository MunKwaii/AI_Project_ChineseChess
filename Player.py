import random
import math
from Chinese_Chess_Game_Rules import ChessGame, PIECES, calculate_absolute_points, _Piece
import numpy as np
import tensorflow as tf
from collections import deque
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
    
    def expand(self, policy=None):
        valid_moves = self.game.get_valid_moves()
        if policy is not None:
            move_probs = policy
            for move, prob in zip(valid_moves, move_probs):
                new_game = self.game.copy_and_make_move(move)
                child = MCTSNode(new_game, move, self)
                child.prior = prob
                self.children.append(child)
        else:
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
    def __init__(self, iterations=500, simulate_fn=None, exploration_weight=1.0):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.move_history = deque(maxlen=10)
        self.simulate_fn = simulate_fn if simulate_fn else self.default_evaluate_board
        self.experience_buffer = deque(maxlen=10000)
        self.opening_moves = {
            'Red': [
                "c2.5",  # Pháo đầu
                "h2+3",  # Mã trái
                "r1+1",  # Xe trái ra
                "h8+7",  # Mã phải
                "r9+1",  # Xe phải ra
            ],
            'Black': [
                "h8+7",  # Mã phải (phản ứng với Pháo đầu của đỏ)
                "c8.5",  # Pháo đầu
                "r9+1",  # Xe phải ra
                "h2+3",  # Mã trái
                "r1+1",  # Xe trái ra
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
    
    def get_move_policy(self, game):
        if self.simulate_fn == self.default_evaluate_board:
            return None
        state_tensor = self.board_to_tensor(game)
        q_values = self.simulate_fn.main_network.predict(state_tensor[np.newaxis, ...], verbose=0)[0]
        exp_q = np.exp(q_values - np.max(q_values))
        return exp_q / exp_q.sum()

    def default_evaluate_board(self, game):
        board = game.get_board()
        total_score = calculate_absolute_points(board)
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece and piece.is_red == game.is_red_move():
                    if piece.kind in {'r', 'c', 'h'}:
                        if (piece.is_red and y < 9) or (not piece.is_red and y > 0):
                            total_score += 50
                    if 4 <= y <= 5 and 3 <= x <= 5:
                        total_score += 20 if piece.kind in {'r', 'c'} else 10
        major_pieces_moved = sum(1 for m in self.move_history if m[0] in {'r', 'c', 'h'})
        if len(self.move_history) < 5 and any(m.startswith('p') for m in self.move_history) and major_pieces_moved < 2:
            total_score -= 30
        current_move = self.move_history[-1] if self.move_history else None
        if current_move:
            if list(self.move_history).count(current_move) >= 2:
                total_score -= 150
            elif list(self.move_history).count(current_move) == 1:
                total_score -= 80
        return min(max(total_score / 10000 * (1 if game.is_red_move() else -1), -1), 1)

    def make_move(self, game, previous_move):
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        move_count = len(self.move_history)
        if move_count < 5:
            side = 'Red' if game.is_red_move() else 'Black'
            opening_move = self.opening_moves[side][move_count] if move_count < len(self.opening_moves[side]) else None
            if opening_move and opening_move in valid_moves:
                self.move_history.append(opening_move)
                return opening_move
        root = MCTSNode(game)
        move_policy = self.get_move_policy(game)
        for _ in range(self.iterations):
            node = root
            while node.children and node.game.get_winner() is None:
                node = max(node.children, key=lambda c: c.ucb1(self.exploration_weight))
            if node.game.get_winner() is None:
                node = node.expand(move_policy)
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
        best_child = max(root.children, key=lambda c: c.visits)
        self.move_history.append(best_child.move)
        return best_child.move

    def get_experience_batch(self, batch_size):
        if len(self.experience_buffer) < batch_size:
            return None
        return random.sample(self.experience_buffer, batch_size)
    
    def reload_tree(self):
        pass

    def save_training_data(self, data_path):
        np.save(data_path, np.array(self.experience_buffer, dtype=object))
    
    def load_training_data(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.experience_buffer = deque(data, maxlen=self.experience_buffer.maxlen)

class DQNAgent:
    def __init__(self, state_size=(10, 9, len(PIECES)), action_size=200):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_targetnn_rate = 100
        self.batch_size = 64
        self.main_network = self.get_nn()
        self.target_network = self.get_nn()
        self.target_network.set_weights(self.main_network.get_weights())
        self.steps_since_update = 0

    def _initialize_buffer(self):
        random_player = RandomPlayer()
        for _ in range(50):
            game = ChessGame()
            state = np.zeros(self.state_size, dtype=np.float32)
            while game.get_winner() is None:
                move = random_player.make_move(game, None)
                game.make_move(move)
                next_state = mcts_player.board_to_tensor(game)
                reward = 0
                if game.get_winner() == 'Red':
                    reward = 1 if game.is_red_move() else -1
                elif game.get_winner() == 'Black':
                    reward = -1 if game.is_red_move() else 1
                self.replay_buffer.append((state, random.randint(0, self.action_size-1), reward, next_state, game.get_winner() is not None))
                state = next_state

    def get_nn(self):
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
        state_batch = np.array([batch[0] for batch in exp_batch])
        action_batch = np.array([batch[1] for batch in exp_batch])
        reward_batch = [batch[2] for batch in exp_batch]
        next_state_batch = np.array([batch[3] for batch in exp_batch])
        terminal_batch = [batch[4] for batch in exp_batch]
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch
    
    def tensor_to_board(self, tensor):
        board = [[None for _ in range(9)] for _ in range(10)]
        piece_keys = list(PIECES.keys())
        if isinstance(tensor, np.ndarray):
            for y in range(10):
                for x in range(9):
                    piece_idx = np.argmax(tensor[y, x])
                    if piece_idx > 0 and piece_idx - 1 < len(piece_keys):
                        kind, is_red = piece_keys[piece_idx - 1]
                        board[y][x] = _Piece(kind, is_red)
        else:
            for y in range(10):
                for x in range(9):
                    piece = tensor[y][x]
                    if piece is not None:
                        board[y][x] = _Piece(piece.kind, piece.is_red)
        return board

    def train_main_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.get_batch_from_buffer(self.batch_size)
        q_values = self.main_network.predict(state_batch, verbose=0)
        next_q_values = self.target_network.predict(next_state_batch, verbose=0)
        max_next_q = np.amax(next_q_values, axis=1)
        for i in range(self.batch_size):
            new_q_values = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * max_next_q[i]
            q_values[i][action_batch[i]] = new_q_values
        self.main_network.fit(state_batch, q_values, batch_size=self.batch_size, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.steps_since_update += 1
        if self.steps_since_update >= self.update_targetnn_rate:
            self.target_network.set_weights(self.main_network.get_weights())
            self.steps_since_update = 0

    def make_decision(self, state, valid_moves, training=True):
        if training and random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_moves)
        state = state[np.newaxis, ...]
        q_values = self.main_network.predict(state, verbose=0)[0]
        valid_indices = [self.encode_move(move, valid_moves) for move in valid_moves]
        valid_q = [q_values[i] for i in valid_indices]
        best_idx = np.argmax(valid_q)
        return valid_moves[best_idx]
    
    def save_model(self, model_path, target_path=None):
        self.main_network.save_weights(model_path)
        if target_path:
            self.target_network.save_weights(target_path)
    
    def load_model(self, model_path, target_path=None):
        self.main_network.load_weights(model_path)
        if target_path:
            self.target_network.load_weights(target_path)
        else:
            self.target_network.set_weights(self.main_network.get_weights())
    
    def save_experience_buffer(self, buffer_path):
        np.save(buffer_path, np.array(self.replay_buffer, dtype=object))
    
    def load_experience_buffer(self, buffer_path):
        buffer = np.load(buffer_path, allow_pickle=True)
        self.replay_buffer = deque(buffer, maxlen=self.replay_buffer.maxlen)

    def save_full_model(self, base_path):
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
            'update_targetnn_rate': self.update_targetnn_rate
        }
        with open(f"{base_path}_params.json", 'w') as f:
            json.dump(params, f)
    
    @classmethod
    def load_full_model(cls, base_path):
        with open(f"{base_path}_params.json", 'r') as f:
            params = json.load(f)
        agent = cls(state_size=(10, 9, len(PIECES)))
        agent.gamma = params['gamma']
        agent.epsilon = params['epsilon']
        agent.epsilon_min = params['epsilon_min']
        agent.epsilon_decay = params['epsilon_decay']
        agent.learning_rate = params['learning_rate']
        agent.batch_size = params['batch_size']
        agent.update_targetnn_rate = params['update_targetnn_rate']
        agent.main_network.load_weights(f"{base_path}_main.weights.h5")
        agent.target_network.load_weights(f"{base_path}_target.weights.h5")
        exp_buffer = np.load(f"{base_path}_exp.npy", allow_pickle=True)
        agent.replay_buffer = deque(exp_buffer, maxlen=agent.replay_buffer.maxlen)
        return agent

if __name__ == '__main__':
    dqn_agent = DQNAgent()
    mcts_player = MCTSPlayer(simulate_fn=dqn_agent, iterations=500)
    dqn_agent._initialize_buffer()
    for i in range(1000):
        game = ChessGame()
        print(f"Vòng {i + 1}")
        state = mcts_player.board_to_tensor(game)
        while game.get_winner() is None:
            valid_moves = game.get_valid_moves()
            move = mcts_player.make_move(game, None)
            action_idx = dqn_agent.encode_move(move, valid_moves)
            reward = 0
            next_game = game.copy_and_make_move(move)
            if next_game.get_winner() == ('Red' if game.is_red_move() else 'Black'):
                reward = 1
            elif next_game.get_winner() == ('Black' if game.is_red_move() else 'Red'):
                reward = -1
            elif move[0] in {'r', 'c', 'h'}:
                reward = 0.05
            elif move.startswith('p') and len(mcts_player.move_history) < 5:
                reward = -0.05
            next_state = mcts_player.board_to_tensor(next_game)
            dqn_agent.save_experience(state, action_idx, reward, next_state, next_game.get_winner() is not None)
            game.make_move(move)
            state = next_state
            if len(dqn_agent.replay_buffer) >= dqn_agent.batch_size:
                dqn_agent.train_main_network()
            print(f"Move: {move}, Epsilon: {dqn_agent.epsilon:.4f}")
            print(game)
        print(f"Winner: {game.get_winner()}")
        if i % 50 == 0:
            dqn_agent.save_full_model('trained_models/chinese_chess_dqn')
            mcts_player.save_training_data('mcts_experience.npy')
    dqn_agent.save_full_model('trained_models/chinese_chess_dqn')
    mcts_player.save_training_data('mcts_experience.npy')
    game = ChessGame()
    while game.get_winner() is None:
        move = mcts_player.make_move(game, None)
        game.make_move(move)
        print(f"Move: {move}")
        print(game)
    print(f"Final winner: {game.get_winner()}") 