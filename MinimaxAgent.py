import logging
from Chinese_Chess_Game_Rules import ChessGame, _MAX_MOVES, _get_index_movement

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('minimax.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class MinimaxAgent:
    def __init__(self, max_depth=3):
        """
        Khởi tạo MinimaxAgent.
        
        Args:
            max_depth (int): Độ sâu tối đa cho thuật toán Minimax.
        """
        self.max_depth = max_depth
        self.nodes_evaluated = 0  

    def get_move(self, game, prev_move=None):
        """
        Tìm nước đi tốt nhất sử dụng thuật toán Minimax.
        
        Args:
            game (ChessGame): Trạng thái hiện tại của trò chơi.
            prev_move (str, optional): Nước đi trước đó (không sử dụng trong phiên bản này).
        
        Returns:
            str: Nước đi tốt nhất ở định dạng WXF, hoặc None nếu không có nước đi hợp lệ.
        """
        self.nodes_evaluated = 0  # Reset bộ đếm
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            logging.info("No valid moves available")
            return None

        # Tính và ghi log thông tin lý thuyết/dự kiến
        branching_factor = len(valid_moves)
        minimax_nodes_theoretical = branching_factor ** self.max_depth
        alpha_beta_nodes_expected = int(branching_factor ** (self.max_depth / 2))
        
        logging.info(f"Branching factor trung bình: ~{branching_factor}.")
        logging.info(f"Số nút duyệt lý thuyết (Minimax, không cắt tỉa): {branching_factor}^{self.max_depth} = {minimax_nodes_theoretical}.")
        logging.info(f"Số nút duyệt dự kiến (Alpha-Beta, với cắt tỉa): ~{branching_factor}^({self.max_depth}/2) ~ {alpha_beta_nodes_expected}.")

        is_red = game.is_red_move()
        best_value = float('-inf') if is_red else float('inf')
        best_move = None

        logging.info(f"Starting Minimax search with depth {self.max_depth}, is_red: {is_red}, valid moves: {len(valid_moves)}")

        # Sắp xếp nước đi để ưu tiên các nước bắt quân
        capture_moves = [move for move in valid_moves if self.is_capture_move(game, move)]
        non_capture_moves = [move for move in valid_moves if move not in capture_moves]
        sorted_moves = capture_moves + non_capture_moves

        for move in sorted_moves:
            new_game = game.copy_and_make_move_alphabeta(move)
            value = self.minimax(new_game, depth=self.max_depth - 1, is_max=not is_red)
            
            if is_red:  # MAX player (Red)
                if value > best_value:
                    best_value = value
                    best_move = move
            else:  # MIN player (Black)
                if value < best_value:
                    best_value = value
                    best_move = move

        logging.info(f"Minimax selected move: {best_move}, value: {best_value}, nodes evaluated: {self.nodes_evaluated}")
        return best_move

    def is_capture_move(self, game, move):
        """
        Kiểm tra xem nước đi có phải là nước bắt quân không.
        
        Args:
            game (ChessGame): Trạng thái hiện tại của trò chơi.
            move (str): Nước đi cần kiểm tra.
        
        Returns:
            bool: True nếu nước đi bắt quân, False nếu không.
        """
        board = game.get_board()
        is_red = game.is_red_move()
        end_pos = _get_index_movement(board, move, is_red)
        target_piece = board[end_pos[0]][end_pos[1]]
        return target_piece is not None and target_piece.is_red != is_red

    def minimax(self, game, depth, is_max):
        """
        Thuật toán Minimax để đánh giá trạng thái trò chơi.
        
        Args:
            game (ChessGame): Trạng thái hiện tại của trò chơi.
            depth (int): Độ sâu còn lại.
            is_max (bool): True nếu là lượt MAX (Red), False nếu là MIN (Black).
        
        Returns:
            float: Giá trị đánh giá của trạng thái.
        """
        self.nodes_evaluated += 1

        # Kiểm tra trạng thái kết thúc hoặc độ sâu
        winner = game.get_winner()
        if winner is not None:
            if winner == 'Red':
                return 10000
            elif winner == 'Black':
                return -10000
            else:  # Draw
                return 0
        if depth == 0:
            return self.evaluate_board(game)

        valid_moves = game.get_valid_moves()
        if not valid_moves:
            if game.is_in_check(game.get_board(), game.is_red_move()):
                return -10000 if is_max else 10000  # Checkmate
            return 0  # Stalemate

        logging.debug(f"Depth {self.max_depth - depth}: {len(valid_moves)} valid moves")

        # Sắp xếp nước đi để ưu tiên các nước bắt quân
        capture_moves = [move for move in valid_moves if self.is_capture_move(game, move)]
        non_capture_moves = [move for move in valid_moves if move not in capture_moves]
        sorted_moves = capture_moves + non_capture_moves

        if is_max:  # MAX player (Red)
            max_value = float('-inf')
            for move in sorted_moves:
                new_game = game.copy_and_make_move_alphabeta(move)
                value = self.minimax(new_game, depth - 1, False)
                max_value = max(max_value, value)
            return max_value
        else:  # MIN player (Black)
            min_value = float('inf')
            for move in sorted_moves:
                new_game = game.copy_and_make_move_alphabeta(move)
                value = self.minimax(new_game, depth - 1, True)
                min_value = min(min_value, value)
            return min_value

    def evaluate_board(self, game):
        """
        Đánh giá trạng thái bàn cờ dựa trên các yếu tố chiến lược.
        
        Args:
            game (ChessGame): Trạng thái hiện tại của trò chơi.
        
        Returns:
            float: Giá trị đánh giá (dương nếu tốt cho Red, âm nếu tốt cho Black).
        """
        board = game.get_board()
        score = 0

        # Giá trị quân cờ
        piece_values = {
            'p': 100,  # Tốt (tăng lên 200 nếu qua sông)
            'a': 200,  # Sĩ
            'e': 200,  # Tượng
            'h': 400,  # Mã
            'c': 450,  # Pháo
            'r': 900,  # Xe
            'k': 10000  # Tướng
        }

        # Đếm quân và tính điểm
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece is None:
                    continue

                side = 1 if piece.is_red else -1
                kind = piece.kind
                base_value = piece_values[kind]

                # Điều chỉnh giá trị Tốt nếu qua sông
                if kind == 'p':
                    if (piece.is_red and y <= 4) or (not piece.is_red and y >= 5):
                        base_value = 200

                score += side * base_value

                # Thưởng vị trí chiến lược
                # Trung tâm: hàng 3-7, cột 3-5
                if 3 <= y <= 7 and 3 <= x <= 5:
                    if kind in ['r', 'c', 'h']:  # Xe, Pháo, Mã
                        score += side * 50  # Thưởng kiểm soát trung tâm

                # Phạt Mã nếu bị kẹt gần biên
                if kind == 'h':
                    if x in [0, 8] or y in [0, 9]:
                        score -= side * 30

                # Thưởng Pháo nếu có quân cản (ngòi)
                if kind == 'c':
                    for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        steps = 1
                        found_screen = False
                        while 0 <= y + dy * steps < 10 and 0 <= x + dx * steps < 9:
                            target = board[y + dy * steps][x + dx * steps]
                            if target:
                                if not found_screen:
                                    found_screen = True
                                else:
                                    if target.is_red != piece.is_red:
                                        score += side * 100  # Thưởng nếu có thể ăn quân
                                    break
                            steps += 1

                # Phạt Tướng nếu ra khỏi cung hoặc bị chiếu
                if kind == 'k':
                    is_in_palace = (7 <= y <= 9 and 3 <= x <= 5) if piece.is_red else (0 <= y <= 2 and 3 <= x <= 5)
                    if not is_in_palace:
                        score -= side * 200
                    if game.is_in_check(board, piece.is_red):
                        score -= side * 500

        # Tính di động: Số nước đi hợp lệ của các quân mạnh (Xe, Pháo, Mã)
        mobility_score = 0
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece and piece.kind in ['r', 'c', 'h']:
                    side = 1 if piece.is_red else -1
                    temp_game = ChessGame(board=board, red_active=piece.is_red)
                    moves = temp_game.get_valid_moves()
                    mobility_score += side * len(moves) * 10  # Thưởng 10 điểm mỗi nước đi

        score += mobility_score

        # Thưởng uy hiếp Tướng đối phương
        if game.is_red_move():
            if game.is_in_check(board, False):  # Red uy hiếp Black
                score += 300
        else:
            if game.is_in_check(board, True):  # Black uy hiếp Red
                score -= 300

        return score

    @staticmethod
    def run_test(max_depth=3, fen=None):
        """
        Chạy thử nghiệm thuật toán Minimax trên một trò chơi cờ tướng.
        Cả hai bên (Red và Black) đều sử dụng Minimax để chơi.
        
        Args:
            max_depth (int): Độ sâu tìm kiếm của Minimax.
            fen (str, optional): Chuỗi FEN để khởi tạo bàn cờ. Nếu None, dùng bàn cờ mặc định.
        """
        # Tạo trò chơi mới
        game = ChessGame()
        if fen:
            try:
                game.set_board_from_fen(fen)
                logging.info(f"Initialized board from FEN: {fen}")
            except ValueError as e:
                logging.error(f"Invalid FEN: {e}")
                return

        # Tạo agent
        agent = MinimaxAgent(max_depth=max_depth)
        move_count = 0

        # In bàn cờ ban đầu
        print("Initial board:")
        print(game)

        # Vòng lặp chơi cờ
        while move_count < _MAX_MOVES:
            # Kiểm tra trạng thái kết thúc
            winner = game.get_winner()
            if winner:
                print(f"Game ended: {winner}")
                break

            # Lấy nước đi từ Minimax
            move = agent.make_move(game)
            if not move:
                print(f"No valid moves available for {'Red' if game.is_red_move() else 'Black'}")
                break

            # Thực hiện nước đi
            print(f"Move {move_count + 1}: {move} ({'Red' if game.is_red_move() else 'Black'})")
            game.make_move(move)
            move_count += 1

            # In bàn cờ sau nước đi
            print(f"Board after move {move_count}:")
            print(game)

        if move_count >= _MAX_MOVES:
            print(f"Stopped after {_MAX_MOVES} moves: Draw")

if __name__ == "__main__":
    MinimaxAgent.run_test(max_depth=15)