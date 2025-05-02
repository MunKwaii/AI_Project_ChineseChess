import logging
from Chinese_Chess_Game_Rules import ChessGame, _MAX_MOVES, _get_index_movement

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alphabeta.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class AlphaBetaAgent:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.nodes_evaluated = 0  # Đếm số nút được đánh giá
    def is_capture_move(self, game, move):
        board = game.get_board()
        is_red = game.is_red_move()
        end_pos = _get_index_movement(board, move, is_red)
        target_piece = board[end_pos[0]][end_pos[1]]
        return target_piece is not None and target_piece.is_red != is_red

    def is_threatening_king(self, game, move):
        board = game.get_board()
        is_red = game.is_red_move()
        end_pos = _get_index_movement(board, move, is_red)
        piece = board[end_pos[0]][end_pos[1]]
        if piece and piece.kind == 'k' and piece.is_red != is_red:
            return True
        return False
    
    def get_move(self, game, prev_move=None):
        """
        Tìm nước đi tốt nhất sử dụng thuật toán Alpha-Beta.
        
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
        alpha_beta_nodes_expected_min = int(branching_factor ** (self.max_depth / 2))
        alpha_beta_nodes_expected_max = branching_factor ** self.max_depth
        logging.info(f"Số nút duyệt dự kiến (Alpha-Beta, với cắt tỉa): từ ~{branching_factor}^({self.max_depth}/2) ~ {alpha_beta_nodes_expected_min} đến {alpha_beta_nodes_expected_max}.")
        
        is_red = game.is_red_move()
        best_value = float('-inf') if is_red else float('inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        logging.info(f"Starting Alpha-Beta search with depth {self.max_depth}, is_red: {is_red}, valid moves: {len(valid_moves)}")

        # Sắp xếp nước đi: Ưu tiên uy hiếp Tướng, sau đó là bắt quân
        threatening_moves = [move for move in valid_moves if self.is_threatening_king(game, move)]
        capture_moves = [move for move in valid_moves if self.is_capture_move(game, move) and move not in threatening_moves]
        non_capture_moves = [move for move in valid_moves if move not in threatening_moves and move not in capture_moves]
        sorted_moves = threatening_moves + capture_moves + non_capture_moves

        for move in sorted_moves:
            new_game = game.copy_and_make_move(move)
            value = self.alpha_beta(new_game, depth=self.max_depth - 1, is_max=not is_red, alpha=alpha, beta=beta)
            
            if is_red:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)

        logging.info(f"Alpha-Beta selected move: {best_move}, value: {best_value}, nodes evaluated: {self.nodes_evaluated}")
        return best_move

    def is_capture_move(self, game, move):
        board = game.get_board()
        is_red = game.is_red_move()
        end_pos = _get_index_movement(board, move, is_red)
        target_piece = board[end_pos[0]][end_pos[1]]
        return target_piece is not None and target_piece.is_red != is_red

    def alpha_beta(self, game, depth, is_max, alpha, beta):
        """
        Thuật toán Alpha-Beta Pruning để đánh giá trạng thái trò chơi.
        
        Args:
            game (ChessGame): Trạng thái hiện tại của trò chơi.
            depth (int): Độ sâu còn lại.
            is_max (bool): True nếu là lượt MAX (Red), False nếu là MIN (Black).
            alpha (float): Giá trị tốt nhất cho MAX.
            beta (float): Giá trị tốt nhất cho MIN.
        
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
                new_game = game.copy_and_make_move(move)
                value = self.alpha_beta(new_game, depth - 1, False, alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break  # Cắt tỉa
            return max_value
        else:  # MIN player (Black)
            min_value = float('inf')
            for move in sorted_moves:
                new_game = game.copy_and_make_move(move)
                value = self.alpha_beta(new_game, depth - 1, True, alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, min_value)
                if beta <= alpha:
                    break  # Cắt tỉa
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
            'p': 100,  # Tốt (sẽ điều chỉnh theo vị trí)
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

                # Điều chỉnh giá trị Tốt
                if kind == 'p':
                    # Qua sông
                    if (piece.is_red and y <= 4) or (not piece.is_red and y >= 5):
                        base_value = 200
                    # Thưởng nếu Tốt tiến gần cung đối phương
                    if piece.is_red and y >= 7:
                        base_value += 50 * (y - 6)  # Thưởng thêm nếu ở hàng 7-9
                    elif not piece.is_red and y <= 2:
                        base_value += 50 * (3 - y)  # Thưởng thêm nếu ở hàng 0-2
                    # Phạt nếu Tốt ở biên và chưa qua sông
                    if x in [0, 8] and ((piece.is_red and y < 5) or (not piece.is_red and y > 4)):
                        base_value -= 20

                # Thưởng Xe ở cột mở và hàng cuối
                if kind == 'r':
                    # Cột mở: Không có Tốt nào trong cột
                    is_open_column = True
                    for row in range(10):
                        if row != y and board[row][x] and board[row][x].kind == 'p':
                            is_open_column = False
                            break
                    if is_open_column:
                        score += side * 50
                    # Hàng cuối đối phương
                    if (piece.is_red and y == 0) or (not piece.is_red and y == 9):
                        score += side * 100

                # Phạt Mã bị chặn chân
                if kind == 'h':
                    # Kiểm tra các vị trí cản (các điểm mà Mã nhảy qua)
                    blocking_positions = [
                        (y - 1, x) if y - 2 >= 0 else None,  # Nhảy lên
                        (y + 1, x) if y + 2 <= 9 else None,  # Nhảy xuống
                        (y, x - 1) if x - 2 >= 0 else None,  # Nhảy trái
                        (y, x + 1) if x + 2 <= 8 else None,  # Nhảy phải
                    ]
                    for pos in blocking_positions:
                        if pos and board[pos[0]][pos[1]]:
                            score -= side * 40  # Phạt nếu bị chặn

                score += side * base_value

                # Thưởng vị trí chiến lược: Trung tâm
                if 3 <= y <= 7 and 3 <= x <= 5:
                    if kind in ['r', 'c', 'h']:
                        score += side * 50

                # Phạt Mã nếu bị kẹt gần biên
                if kind == 'h':
                    if x in [0, 8] or y in [0, 9]:
                        score -= side * 30

                # Thưởng Pháo nếu có ngòi
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
                                        score += side * 100
                                    break
                            steps += 1

                # Phạt Tướng nếu ra khỏi cung hoặc bị chiếu
                if kind == 'k':
                    is_in_palace = (7 <= y <= 9 and 3 <= x <= 5) if piece.is_red else (0 <= y <= 2 and 3 <= x <= 5)
                    if not is_in_palace:
                        score -= side * 200
                    if game.is_in_check(board, piece.is_red):
                        score -= side * 500

        # Tính di động: Số nước đi hợp lệ của các quân mạnh
        mobility_score = 0
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece and piece.kind in ['r', 'c', 'h']:
                    side = 1 if piece.is_red else -1
                    temp_game = ChessGame(board=board, red_active=piece.is_red)
                    moves = temp_game.get_valid_moves()
                    mobility_score += side * len(moves) * 10

        score += mobility_score

        # Thưởng uy hiếp Tướng đối phương
        if game.is_red_move():
            if game.is_in_check(board, False):  # Red uy hiếp Black
                score += 300
        else:
            if game.is_in_check(board, True):  # Black uy hiếp Red
                score -= 300

        # Thưởng nếu Tướng đối mặt
        red_king_pos = None
        black_king_pos = None
        for y in range(10):
            for x in range(9):
                piece = board[y][x]
                if piece and piece.kind == 'k':
                    if piece.is_red:
                        red_king_pos = (y, x)
                    else:
                        black_king_pos = (y, x)

        if red_king_pos and black_king_pos and red_king_pos[1] == black_king_pos[1]:  # Cùng cột
            is_blocked = False
            for row in range(min(red_king_pos[0], black_king_pos[0]) + 1, max(red_king_pos[0], black_king_pos[0])):
                if board[row][red_king_pos[1]]:
                    is_blocked = True
                    break
            if not is_blocked:
                score += 500 if game.is_red_move() else -500  # Thưởng/phạt nếu Tướng đối mặt

        # Phạt nếu cung Tướng yếu (thiếu Sĩ/Tượng)
        for is_red in [True, False]:
            side = 1 if is_red else -1
            advisors = 0  # Sĩ
            elephants = 0  # Tượng
            for y in range(10):
                for x in range(9):
                    piece = board[y][x]
                    if piece and piece.is_red == is_red:
                        if piece.kind == 'a':
                            advisors += 1
                        elif piece.kind == 'e':
                            elephants += 1
            # Phạt nếu thiếu Sĩ/Tượng
            if advisors < 2:
                score -= side * 100 * (2 - advisors)  # Thiếu 1 Sĩ: -100, thiếu 2 Sĩ: -200
            if elephants < 2:
                score -= side * 80 * (2 - elephants)  # Thiếu 1 Tượng: -80, thiếu 2 Tượng: -160

        # Thưởng phối hợp quân (Pháo và Xe cùng kiểm soát cột)
        for col in range(9):
            rook_present = False
            cannon_present = False
            for row in range(10):
                piece = board[row][col]
                if piece:
                    if piece.kind == 'r' and piece.is_red == game.is_red_move():
                        rook_present = True
                    elif piece.kind == 'c' and piece.is_red == game.is_red_move():
                        cannon_present = True
            if rook_present and cannon_present:
                score += 150 if game.is_red_move() else -150  # Thưởng phối hợp

        return score

    @staticmethod
    def run_test(max_depth=3, fen=None):
        game = ChessGame()
        if fen:
            try:
                game.set_board_from_fen(fen)
                logging.info(f"Initialized board from FEN: {fen}")
            except ValueError as e:
                logging.error(f"Invalid FEN: {e}")
                return

        # Tạo agent
        agent = AlphaBetaAgent(max_depth=max_depth)
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

            # Lấy nước đi từ Alpha-Beta
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
    AlphaBetaAgent.run_test(max_depth=3)