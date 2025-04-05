import random
import math 
from Chinese_Chess_Game_Rules import ChessGame

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
        self.game = game # tran thai ban co
        self.move = move # nuoc di dan den tang thai nay
        self.parent = parent #node cha
        self.children = [] #cac node con
        self.visits = 0 #so lan tham
        self.value = 0.0 # tong gia tri

    #cong thuc ucb1
    def ucb1(self):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + 22 *math.sqrt(math.log(self.parent.visits) / self.visits)
    
    #Mo rong
    def expand(self):
        valid_moves = self.game.get_valid_moves()
        for move in valid_moves:
            new_game = self.game.copy_and_make_move(move)
            self.children.append(MCTSNode(new_game, move, self))
        if self.children:
            return self.children[0]
        else:
            return None

    #cap nhap nguoc
    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(-value)

class MCTSPlayer:
    def __init__(self, iterations = 100, simulate_fn = None):
        self.iterations = iterations
        #dung heuristic tam thay cho dqn 
        if simulate_fn:
            self.simulate_fn = simulate_fn
        else:
            self.simulate_fn = self.default_evaluate_board
        self.experience = [] #luu du lieu de train DQN - game, move, value

    # Hàm đánh giá mặc định (heuristic) khi chưa có DQN
    def default_evaluate_board(self, game):
        """Đánh giá bàn cờ dựa trên số quân và vị trí"""
        board = game.get_board()
        red_score = 0  # Điểm của bên đỏ
        black_score = 0  # Điểm của bên đen
        
        # Giá trị cơ bản của các quân cờ (dựa trên cờ tướng truyền thống)
        piece_values = {'r': 9, 'h': 4, 'e': 2, 'a': 2, 'k': 100, 'c': 4.5, 'p': 1}
        
        # Tính điểm cho từng ô trên bàn cờ
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece:
                    value = piece_values[piece.kind]  # Lấy giá trị quân
                    if piece.is_red:
                        red_score += value
                        # Thưởng thêm nếu pháo ở giữa bàn (cột 3, 4, 5)
                        if piece.kind == 'c' and x in [3, 4, 5]:
                            red_score += 0.5
                    else:
                        black_score += value
                        if piece.kind == 'c' and x in [3, 4, 5]:
                            black_score += 0.5
        
        # Tính điểm chênh lệch và chuẩn hóa về [-1, 1]
        total_score = red_score - black_score
        if game.is_red_move():  # Nếu là lượt đỏ, ưu tiên điểm đỏ
            return min(max(total_score / 100, -1), 1)
        else:  # Nếu là lượt đen, ưu tiên điểm đen
            return min(max(-total_score / 100, -1), 1)

    def make_move(self, game, previous_move):
        root = MCTSNode(game)
        if game.get_winner() is not None: # van co ket thuc 
            valid_moves = game.get_valid_moves()
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return None
        valid_moves = game.get_valid_moves()

        if not valid_moves:
            return None
    
        for _ in range(self.iterations):
            node = root 
            #1. Selection
            while node.children and node.game.get_winner() is None:
                node = max(node.children, key = lambda c: c.ucb1())
            #2. Expansion
            if node.game.get_winner() is None:
                node = node.expand()
            if node:
                #3. Simulation
                value = self.simulate_fn(node.game)
                #4. Backpropagation
                node.backpropagate(value)

                self.experience.append((node.game, node.move, value))
            else:
                break

        #chon nuoc di duoc tham nhieu nhat
        if not root.children:
            if valid_moves:
                return random.choice(valid_moves)
            else:
                return None
        best_child = max(root.children, key = lambda c: c.visits)
        return best_child.move
    
    # du lieu train dqn
    def get_experience(self):
        return self.experience
    

   
    


        
# import random
# import math
# from Chinese_Chess_Game_Rules import ChessGame

# class Player:
#     def make_move(self, game, previous_move):
#         raise NotImplementedError("Phải triển khai make_move trong lớp con")

#     def reload_tree(self):
#         raise NotImplementedError("Phải triển khai reload_tree trong lớp con")

# class MCTSNode:
#     def __init__(self, game, move=None, parent=None):
#         self.game = game # tran thai ban co
#         self.move = move # nuoc di dan den tang thai nay
#         self.parent = parent #node cha
#         self.children = [] #cac node con
#         self.visits = 0 #so lan tham
#         self.value = 0.0 # tong gia tri

#     #cong thuc ucb1
#     def ucb1(self):
#         if self.visits == 0:
#             return float('inf')
#         return (self.value / self.visits) + 2 * math.sqrt(math.log(self.parent.visits) / self.visits)
    
#     def expand(self):
#         valid_moves = self.game.get_valid_moves()
#         if not valid_moves:  # neu k co nc di naonao
#             return None
#         for move in valid_moves:
#             new_game = self.game.copy_and_make_move(move)
#             self.children.append(MCTSNode(new_game, move, self))
#         return self.children[0]  # Tra ve node con dau tientien

#     def simulate(self):
#         current = self.game
#         while current.get_winner() is None:
#             moves = current.get_valid_moves()
#             if not moves:
#                 break
#             move = random.choice(moves)
#             current = current.copy_and_make_move(move)
#         winner = current.get_winner()
#         if winner == 'Red':
#             return 1 if self.game.is_red_move() else -1
#         elif winner == 'Black':
#             return -1 if self.game.is_red_move() else 1
#         return 0  # hoahoa
    
#     #cap nhap nguoc
#     def backpropagate(self, value):
#         self.visits += 1
#         self.value += value
#         if self.parent:
#             self.parent.backpropagate(-value)

# class MCTSPlayer(Player):  # Kế thừa từ Player
#     def __init__(self, iterations=100):
#         self.iterations = iterations

#     def make_move(self, game, previous_move):
#         root = MCTSNode(game)
#         if game.get_winner() is not None:
#             return None
        
#         valid_moves = game.get_valid_moves()
#         if not valid_moves:
#             return None
        
#         for _ in range(self.iterations):
#             node = root 
#             while node.children and node.game.get_winner() is None:
#                 node = max(node.children, key=lambda c: c.ucb1())
#             if node.game.get_winner() is None:
#                 node = node.expand()
#             if node:
#                 value = node.simulate()
#                 node.backpropagate(value)
        
#         if not root.children:
#             return random.choice(valid_moves) if valid_moves else None
        
#         best_child = max(root.children, key=lambda c: c.visits)
#         return best_child.move
   