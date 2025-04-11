import statistics
import copy

_MAX_MOVES = 200

RED = '\033[91m'
BLACK = '\33[0m'

PIECES = {('r', True): RED + '车' + BLACK, ('r', False): '车',
          ('h', True): RED + '马' + BLACK, ('h', False): '马',
          ('e', True): RED + '相' + BLACK, ('e', False): '象',
          ('a', True): RED + '仕' + BLACK, ('a', False): '士',
          ('k', True): RED + '帅' + BLACK, ('k', False): '将',
          ('c', True): RED + '炮' + BLACK, ('c', False): '炮',
          ('p', True): RED + '兵' + BLACK, ('p', False): '卒'}

class ChessGame:
    def __init__(self, board=None, red_active=True, move_count=0):
        if board is not None:
            self._board = board
        else:
            self._board = [
                [_Piece('r', False), _Piece('h', False), _Piece('e', False), _Piece('a', False), _Piece('k', False), _Piece('a', False), _Piece('e', False), _Piece('h', False), _Piece('r', False)],
                [None for _ in range(0, 9)],
                [None, _Piece('c', False), None, None, None, None, None, _Piece('c', False), None],
                [_Piece('p', False), None, _Piece('p', False), None, _Piece('p', False), None, _Piece('p', False), None, _Piece('p', False)],
                [None for _ in range(0, 9)],
                [None for _ in range(0, 9)],
                [_Piece('p', True), None, _Piece('p', True), None, _Piece('p', True), None, _Piece('p', True), None, _Piece('p', True)],
                [None, _Piece('c', True), None, None, None, None, None, _Piece('c', True), None],
                [None for _ in range(0, 9)],
                [_Piece('r', True), _Piece('h', True), _Piece('e', True), _Piece('a', True), _Piece('k', True), _Piece('a', True), _Piece('e', True), _Piece('h', True), _Piece('r', True)],
            ]

        self._is_red_active = red_active
        self._move_count = move_count
        self._valid_moves = []

        self.move_history = []  # Thêm lịch sử nước đi
        self.last_move = None   # Lưu nước đi gần nhất

        self._recalculate_valid_moves()
        self.board_history = {}  # Thêm dòng này để lưu trạng thái bàn cờ

    def __str__(self):
        print_board(self._board)
        winner = self.get_winner()

        if winner is None:
            turn_message = ''
            if self._is_red_active:
                turn_message += "Red's turn.\n"
            else:
                turn_message += "Black's turn.\n"
            return turn_message + f'Valid moves: {self._valid_moves}'
        elif winner == 'Draw':
            return 'Draw!'
        else:
            return f'{winner} wins!'

    def get_valid_moves(self):
        return self._valid_moves

    def make_move(self, move):
        move_lowered = move.lower()

        if move_lowered not in self._valid_moves:
            raise ValueError(f'Move "{move}" is not valid')

        self._board = self._board_after_move(move_lowered, self._is_red_active)

        # Lưu trạng thái bàn cờ
        board_state = tuple(tuple((cell.kind, cell.is_red) if cell else None for cell in row) for row in self._board)
        self.board_history[board_state] = self.board_history.get(board_state, 0) + 1

        self._is_red_active = not self._is_red_active
        self._move_count += 1
        self._recalculate_valid_moves()
        self.last_move = move
        self.move_history.append(move)

    def copy_and_make_move(self, move):
        new_board = self._board_after_move(move, self._is_red_active)
        new_game = ChessGame(board=new_board,
                             red_active=not self._is_red_active,
                             move_count=self._move_count + 1)
        new_game._valid_moves = []
        return new_game

    def is_red_move(self):
        return self._is_red_active

    def is_checkmate(self):
        if not self.is_in_check(self._board, self._is_red_active):
            return False

        for move in self._valid_moves:
            new_game = self.copy_and_make_move(move)
            if not new_game.is_in_check(new_game._board, self._is_red_active):
                return False
        return True

    def is_in_check(self, board, is_red):
        king_pos = None
        king_piece = _Piece('k', is_red)
        for y in range(10):
            for x in range(9):
                if board[y][x] == king_piece:  # Dùng board thay vì self._board
                    king_pos = (y, x)
                    break
            if king_pos is not None:
                break

        if king_pos is None:
            return False

        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece and piece.is_red != is_red:
                    if self._can_piece_capture_king(board, (r, c), king_pos, piece.kind):
                        return True
        return False

    def is_stalemate(self):
        """Kiểm tra xem có hết nước đi (stalemate) không"""
        if self.is_in_check(self._board, self._is_red_active):
            return False

        # Kiểm tra xem có nước đi hợp lệ nào không
        return len(self._valid_moves) == 0

    def get_winner(self):
        # Kiểm tra lặp lại trạng thái
        board_state = tuple(tuple((cell.kind, cell.is_red) if cell else None for cell in row) for row in self._board)
        if self.board_history.get(board_state, 0) >= 3:
            return 'Draw'  # Hòa do lặp lại 3 lần

        if self._move_count >= _MAX_MOVES:
            return 'Draw'
        elif all(self._board[y][x] != _Piece('k', True)
                for y in range(0, 10) for x in range(0, 9)):
            return 'Black'
        elif all(self._board[y][x] != _Piece('k', False)
                for y in range(0, 10) for x in range(0, 9)):
            return 'Red'
        elif self.is_checkmate():
            return 'Black' if self._is_red_active else 'Red'
        elif self.is_stalemate():
            return 'Draw'
        else:
            return None

    def _calculate_moves_for_board(self, board, is_red_active):
        moves = []

        for pos in [(y, x) for y in range(0, 10) for x in range(0, 9)]:
            piece = board[pos[0]][pos[1]]
            if piece is None or piece.is_red != is_red_active:
                continue

            if piece.kind == 'r':
                moves += self._calculate_moves_for_chariot(board, pos)
            elif piece.kind == 'h':
                moves += self._calculate_moves_for_horse(board, pos)
            elif piece.kind == 'e':
                moves += self._calculate_moves_for_elephant(board, pos)
            elif piece.kind == 'a':
                moves += self._calculate_moves_for_advisor(board, pos)
            elif piece.kind == 'k':
                moves += self._calculate_moves_for_king(board, pos)
            elif piece.kind == 'c':
                moves += self._calculate_moves_for_cannon(board, pos)
            else:
                moves += self._calculate_moves_for_pawn(board, pos)

        return moves

    def _calculate_moves_for_chariot(self, board, pos):
        piece = board[pos[0]][pos[1]]
        moves = []

        moves += self._find_moves_in_direction(board, pos, piece.is_red, (1, 0))
        moves += self._find_moves_in_direction(board, pos, piece.is_red, (-1, 0))
        moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, 1))
        moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, -1))

        return moves

    def _calculate_moves_for_horse(self, board, pos):
        piece = board[pos[0]][pos[1]]
        moves = []

        if pos[0] != 0 and board[pos[0] - 1][pos[1]] is None:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (-2, 1), limit=1)
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (-2, -1), limit=1)
        if pos[0] != 9 and board[pos[0] + 1][pos[1]] is None:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (2, 1), limit=1)
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (2, -1), limit=1)
        if pos[1] != 0 and board[pos[0]][pos[1] - 1] is None:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (1, -2), limit=1)
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (-1, -2), limit=1)
        if pos[1] != 8 and board[pos[0]][pos[1] + 1] is None:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (1, 2), limit=1)
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (-1, 2), limit=1)

        return moves

    def _calculate_moves_for_elephant(self, board, pos):
        piece = board[pos[0]][pos[1]]
        moves = []

        if pos[0] not in {0, 5}:
            if pos[1] != 0 and board[pos[0] - 1][pos[1] - 1] is None:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (-2, -2), limit=1)
            if pos[1] != 8 and board[pos[0] - 1][pos[1] + 1] is None:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (-2, 2), limit=1)
        if pos[0] not in {4, 9}:
            if pos[1] != 0 and board[pos[0] + 1][pos[1] - 1] is None:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (2, -2), limit=1)
            if pos[1] != 8 and board[pos[0] + 1][pos[1] + 1] is None:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (2, 2), limit=1)

        return moves

    def _calculate_moves_for_advisor(self, board, pos):
        piece = board[pos[0]][pos[1]]
        moves = []

        if pos[0] not in {2, 9}:
            if pos[1] != 3:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (1, -1), limit=1)
            if pos[1] != 5:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (1, 1), limit=1)
        if pos[0] not in {0, 7}:
            if pos[1] != 3:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (-1, -1), limit=1)
            if pos[1] != 5:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (-1, 1), limit=1)

        return moves

    def _calculate_moves_for_king(self, board, pos):
        piece = board[pos[0]][pos[1]]
        moves = []

        if pos[0] not in {2, 9}:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (1, 0), limit=1)
        if pos[0] not in {0, 7}:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (-1, 0), limit=1)
        if pos[1] != 3:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, -1), limit=1)
        if pos[1] != 5:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, 1), limit=1)
        if piece.is_red:
            king_row = pos[0] - 1
            while king_row >= 0:
                if board[king_row][pos[1]] is not None and board[king_row][pos[1]].kind == 'k':
                    moves.append(_get_wxf_movement(board, pos, (king_row, pos[1]), piece.is_red))
                elif board[king_row][pos[1]] is not None:
                    break
                king_row -= 1
        else:
            king_row = pos[0] + 1
            while king_row <= 9:
                if board[king_row][pos[1]] is not None and board[king_row][pos[1]].kind == 'k':
                    moves.append(_get_wxf_movement(board, pos, (king_row, pos[1]), piece.is_red))
                elif board[king_row][pos[1]] is not None:
                    break
                king_row += 1

        return moves

    def _calculate_moves_for_cannon(self, board, pos):
        piece = board[pos[0]][pos[1]]
        moves = []

        moves += self._find_moves_in_direction(board, pos, piece.is_red, (1, 0), capture=False)
        moves += self._find_moves_in_direction(board, pos, piece.is_red, (-1, 0), capture=False)
        moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, 1), capture=False)
        moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, -1), capture=False)

        return moves

    def _calculate_moves_for_pawn(self, board, pos):
        piece = board[pos[0]][pos[1]]
        moves = []

        if piece.is_red:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (-1, 0), limit=1)
            if pos[0] <= 4:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, 1), limit=1)
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, -1), limit=1)
        else:
            moves += self._find_moves_in_direction(board, pos, piece.is_red, (1, 0), limit=1)
            if pos[0] >= 5:
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, 1), limit=1)
                moves += self._find_moves_in_direction(board, pos, piece.is_red, (0, -1), limit=1)

        return moves

    def _find_moves_in_direction(self, board, pos, is_red, direction, limit=None, capture=None):
        moves = []
        kind = board[pos[0]][pos[1]].kind
        stop = False
        i = 1
        while not stop:
            y, x = pos[0] + direction[0] * i, pos[1] + direction[1] * i

            if x < 0 or y < 0 or x > 8 or y > 9:
                break

            contents = board[y][x]

            move = _get_wxf_movement(board, pos, (y, x), is_red)

            if contents is not None:
                stop = True
                if kind == 'c':
                    moves += self._check_cannon_fire(board, pos, is_red, direction, i)

                if contents.is_red != is_red and capture is not False:
                    moves.append(move)
            else:
                moves.append(move)

            i += 1

            if limit is not None and i > limit:
                stop = True

        return moves

    def _check_cannon_fire(self, board, pos, is_red, direction, i):
        i += 1
        y, x = pos[0] + direction[0] * i, pos[1] + direction[1] * i
        while 0 <= x <= 8 and 0 <= y <= 9:
            if board[y][x] is not None and board[y][x].is_red == is_red:
                return []
            if board[y][x] is not None and board[y][x].is_red != is_red:
                return [_get_wxf_movement(board, pos, (y, x), is_red)]
            i += 1
            y, x = pos[0] + direction[0] * i, pos[1] + direction[1] * i

        return []

    def get_board(self):
        return self._board

    def _board_after_move(self, move, is_red):
        board_copy = copy.deepcopy(self._board)

        start_pos = _wxf_to_index(self._board, move[0:2], is_red)
        end_pos = _get_index_movement(self._board, move, is_red)

        board_copy[end_pos[0]][end_pos[1]] = board_copy[start_pos[0]][start_pos[1]]
        board_copy[start_pos[0]][start_pos[1]] = None

        return board_copy

    def _get_basic_moves(self, pos, piece_kind, is_red):
        """Tính toán nước đi cơ bản không kiểm tra chiếu"""
        y, x = pos
        moves = []
        piece = self._board[y][x]

        if piece_kind == 'r':  # Xe
            for dy, dx in [(1,0), (-1,0), (0,1), (0,-1)]:
                for i in range(1, 10):
                    ny, nx = y + dy*i, x + dx*i
                    if not (0 <= ny < 10 and 0 <= nx < 9):
                        break
                    target = self._board[ny][nx]
                    if target is None or target.is_red != piece.is_red:
                        moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
                    if target is not None:
                        break

        elif piece_kind == 'h':  # Mã
            horse_moves = [
                (-2,-1), (-2,1), (-1,-2), (-1,2),
                (1,-2), (1,2), (2,-1), (2,1)
            ]
            for dy, dx in horse_moves:
                ny, nx = y + dy, x + dx
                if 0 <= ny < 10 and 0 <= nx < 9:
                    if abs(dy) == 2 and self._board[y + dy//2][x] is not None:
                        continue
                    if abs(dx) == 2 and self._board[y][x + dx//2] is not None:
                        continue
                    target = self._board[ny][nx]
                    if target is None or target.is_red != piece.is_red:
                        moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))

        elif piece_kind == 'e':  # Tượng
            for dy, dx in [(-2,-2), (-2,2), (2,-2), (2,2)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < 10 and 0 <= nx < 9:
                    if self._board[y + dy//2][x + dx//2] is None:
                        if (is_red and ny >= 5) or (not is_red and ny <= 4):
                            target = self._board[ny][nx]
                            if target is None or target.is_red != is_red:
                                moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))

        elif piece_kind == 'a':  # Sĩ
            advisor_moves = [(-1,-1), (-1,1), (1,-1), (1,1)]
            for dy, dx in advisor_moves:
                ny, nx = y + dy, x + dx
                if self._is_in_palace(ny, nx, is_red):
                    target = self._board[ny][nx]
                    if target is None or target.is_red != piece.is_red:
                        moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))

        elif piece_kind == 'k':  # Tướng
            king_moves = [(1,0), (-1,0), (0,1), (0,-1)]
            for dy, dx in king_moves:
                ny, nx = y + dy, x + dx
                if self._is_in_palace(ny, nx, is_red):
                    target = self._board[ny][nx]
                    if target is None or target.is_red != piece.is_red:
                        moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
            self._check_face_to_face(pos, moves, is_red)

        elif piece_kind == 'c':  # Pháo
            for dy, dx in [(1,0), (-1,0), (0,1), (0,-1)]:
                has_screen = False
                for i in range(1, 10):
                    ny, nx = y + dy*i, x + dx*i
                    if not (0 <= ny < 10 and 0 <= nx < 9):
                        break
                    target = self._board[ny][nx]
                    if not has_screen:
                        if target is None:
                            moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
                        else:
                            has_screen = True
                    else:
                        if target is not None:
                            if target.is_red != piece.is_red:
                                moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
                            break

        elif piece_kind == 'p':  # Tốt
            if is_red:
                if y > 0:
                    ny, nx = y-1, x
                    target = self._board[ny][nx]
                    if target is None or target.is_red != piece.is_red:
                        moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
                if y <= 4:
                    if x > 0:
                        ny, nx = y, x-1
                        target = self._board[ny][nx]
                        if target is None or target.is_red != piece.is_red:
                            moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
                    if x < 8:
                        ny, nx = y, x+1
                        target = self._board[ny][nx]
                        if target is None or target.is_red != piece.is_red:
                            moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
            else:
                if y < 9:
                    ny, nx = y+1, x
                    target = self._board[ny][nx]
                    if target is None or target.is_red != piece.is_red:
                        moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
                if y >= 5:
                    if x > 0:
                        ny, nx = y, x-1
                        target = self._board[ny][nx]
                        if target is None or target.is_red != piece.is_red:
                            moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
                    if x < 8:
                        ny, nx = y, x+1
                        target = self._board[ny][nx]
                        if target is None or target.is_red != piece.is_red:
                            moves.append(_get_wxf_movement(self._board, pos, (ny, nx), is_red))
        return moves

    def _is_in_palace(self, y, x, is_red):
        if is_red:
            return 7 <= y <= 9 and 3 <= x <= 5  # Cung của tướng đỏ: hàng 7-9, cột 3-5
        else:
            return 0 <= y <= 2 and 3 <= x <= 5  # Cung của tướng đen: hàng 0-2, cột 3-5

    def _check_face_to_face(self, pos, moves, is_red):
        """Kiểm tra nước đi đối mặt tướng"""
        y, x = pos
        king_piece = _Piece('k', is_red)
        opponent_king = _Piece('k', not is_red)

        # Tìm tướng đối phương
        for ny in range(10):
            if self._board[ny][x] == opponent_king:
                # Kiểm tra không có quân cản
                face_to_face = True
                step = 1 if ny > y else -1
                for i in range(y + step, ny, step):
                    if self._board[i][x] is not None:
                        face_to_face = False
                        break
                if face_to_face:
                    moves.append(_get_wxf_movement(self._board, pos, (ny, x), is_red))
                break

    def calculate_opponent_moves(self):
        raw_moves = []
        for y in range(10):
            for x in range(9):
                piece = self._board[y][x]
                if piece and piece.is_red != self._is_red_active:
                    raw_moves += self._get_basic_moves((y, x), piece.kind, not self._is_red_active)

        # Lọc nước đi hợp lệ
        valid_moves = []
        for move in raw_moves:
            temp_board = self._board_after_move(move, not self._is_red_active)
            if not self.is_in_check(temp_board, not self._is_red_active):
                valid_moves.append(move)

        return valid_moves

    def _recalculate_valid_moves(self):
        raw_moves = []
        for y in range(10):
            for x in range(9):
                piece = self._board[y][x]
                if piece and piece.is_red == self._is_red_active:
                    raw_moves += self._get_basic_moves((y, x), piece.kind, self._is_red_active)

        # Lọc nước đi hợp lệ
        valid_moves = []
        for move in raw_moves:
            temp_board = self._board_after_move(move, self._is_red_active)
            if not self.is_in_check(temp_board, self._is_red_active):
                valid_moves.append(move)

        self._valid_moves = valid_moves

    def _can_piece_capture_king(self, board, piece_pos, king_pos, piece_kind):
        """Kiểm tra 1 quân cờ có thể ăn tướng không."""
        if piece_kind == 'r':  # Xe
            return self._can_chariot_capture(board, piece_pos, king_pos)
        elif piece_kind == 'h':  # Mã
            return self._can_horse_capture(board, piece_pos, king_pos)
        elif piece_kind == 'e':  # Tượng
            return self._can_elephant_capture(board, piece_pos, king_pos)
        elif piece_kind == 'a':  # Sĩ
            return self._can_advisor_capture(board, piece_pos, king_pos)
        elif piece_kind == 'k':  # Tướng
            return self._can_king_capture(board, piece_pos, king_pos)
        elif piece_kind == 'c':  # Pháo
            return self._can_cannon_capture(board, piece_pos, king_pos)
        elif piece_kind == 'p':  # Tốt
            return self._can_pawn_capture(board, piece_pos, king_pos)
        return False

    def _can_chariot_capture(self, board, chariot_pos, king_pos):
        """Kiểm tra xe có thể bắt tướng."""
        chariot_y, chariot_x = chariot_pos
        king_y, king_x = king_pos

        if chariot_y != king_y and chariot_x != king_x:
            return False

        if chariot_y == king_y:  # Cùng hàng
            step = 1 if king_x > chariot_x else -1
            current_x = chariot_x + step
            while current_x != king_x:
                if board[chariot_y][current_x] is not None:
                    return False
                current_x += step
        else:  # Cùng cột
            step = 1 if king_y > chariot_y else -1
            current_y = chariot_y + step
            while current_y != king_y:
                if board[current_y][chariot_x] is not None:
                    return False
                current_y += step
        return True

    def _can_horse_capture(self, board, horse_pos, king_pos):
        """Kiểm tra mã có thể bắt tướng."""
        horse_y, horse_x = horse_pos
        king_y, king_x = king_pos

        # 8 vị trí mã có thể nhảy
        moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]

        for dy, dx in moves:
            new_y, new_x = horse_y + dy, horse_x + dx
            if (new_y, new_x) == king_pos:
                # Kiểm tra chân mã bị cản
                if dy in (-2, 2):
                    block_y = horse_y + (dy // 2)
                    if board[block_y][horse_x] is None:
                        return True
                else:  # dx in (-2, 2)
                    block_x = horse_x + (dx // 2)
                    if board[horse_y][block_x] is None:
                        return True
        return False

    def _can_elephant_capture(self, board, elephant_pos, king_pos):
        """Kiểm tra tượng có thể bắt tướng."""
        elephant_y, elephant_x = elephant_pos
        king_y, king_x = king_pos

        # Tượng chỉ đi chéo 2 ô
        if abs(elephant_y - king_y) == 2 and abs(elephant_x - king_x) == 2:
            # Kiểm tra chân tượng (ô giữa)
            block_y = (elephant_y + king_y) // 2
            block_x = (elephant_x + king_x) // 2
            if board[block_y][block_x] is None:
                # Kiểm tra tượng không vượt sông
                if (elephant_y < 5 and king_y < 5) or (elephant_y >= 5 and king_y >= 5):
                    return True
        return False

    def _can_advisor_capture(self, board, advisor_pos, king_pos):
        """Kiểm tra sĩ có thể bắt tướng."""
        advisor_y, advisor_x = advisor_pos
        king_y, king_x = king_pos

        # Sĩ chỉ đi chéo 1 ô trong cung
        if abs(advisor_y - king_y) == 1 and abs(advisor_x - king_x) == 1:
            # Kiểm tra vẫn ở trong cung
            if 7 <= advisor_y <= 9 and 3 <= advisor_x <= 5:  # Cung đen
                return True
            elif 0 <= advisor_y <= 2 and 3 <= advisor_x <= 5:  # Cung đỏ
                return True
        return False

    def _can_king_capture(self, board, king_pos, other_king_pos):
        """Kiểm tra tướng có thể bắt tướng đối phương (đối diện)."""
        king_y, king_x = king_pos
        other_y, other_x = other_king_pos

        # Hai tướng phải cùng cột và không có quân cản
        if king_x == other_x:
            step = 1 if other_y > king_y else -1
            current_y = king_y + step
            while current_y != other_y:
                if board[current_y][king_x] is not None:
                    return False
                current_y += step
            return True
        return False

    def _can_cannon_capture(self, board, cannon_pos, king_pos):
        """Kiểm tra pháo có thể bắt tướng."""
        cannon_y, cannon_x = cannon_pos
        king_y, king_x = king_pos

        if cannon_y != king_y and cannon_x != king_x:
            return False

        count = 0  # Đếm số quân cản

        if cannon_y == king_y:  # Cùng hàng
            step = 1 if king_x > cannon_x else -1
            current_x = cannon_x + step
            while current_x != king_x:
                if board[cannon_y][current_x] is not None:
                    count += 1
                current_x += step
        else:  # Cùng cột
            step = 1 if king_y > cannon_y else -1
            current_y = cannon_y + step
            while current_y != king_y:
                if board[current_y][cannon_x] is not None:
                    count += 1
                current_y += step

        # Pháo cần đúng 1 quân cản (ngòi)
        return count == 1

    def _can_pawn_capture(self, board, pawn_pos, king_pos):
        """Kiểm tra tốt có thể bắt tướng."""
        pawn_y, pawn_x = pawn_pos
        king_y, king_x = king_pos

        # Tốt đi thẳng, khi qua sông có thể đi ngang
        if board[pawn_y][pawn_x].is_red:  # Tốt đỏ
            # Đi lên (đối với tốt đỏ)
            if (pawn_y - 1 == king_y and pawn_x == king_x) or \
               (pawn_y <= 4 and pawn_y == king_y and abs(pawn_x - king_x) == 1):
                return True
        else:  # Tốt đen
            # Đi xuống (đối với tốt đen)
            if (pawn_y + 1 == king_y and pawn_x == king_x) or \
               (pawn_y >= 5 and pawn_y == king_y and abs(pawn_x - king_x) == 1):
                return True
        return False

def print_board(board):
    row_normal = '丨    丨    丨    丨    丨    丨    丨    丨    丨'
    row_down = '丨    丨    丨    丨 \  丨  / 丨    丨    丨    丨'
    row_up = '丨    丨    丨    丨 /  丨  \ 丨    丨    丨    丨'
    print('一    二    三    四    五    六    七    八    九   (Black)\n'
          f'{_print_row(board, 0)}\n'
          f'{row_down}\n'
          f'{_print_row(board, 1)}\n'
          f'{row_up}\n'
          f'{_print_row(board, 2)}\n'
          f'{row_normal}\n'
          f'{_print_row(board, 3)}\n'
          f'{row_normal}\n'
          f'{_print_row(board, 4)}\n'
          '丨    一    楚    河    一    汉    界    一    丨\n'
          f'{_print_row(board, 5)}\n'
          f'{row_normal}\n'
          f'{_print_row(board, 6)}\n'
          f'{row_normal}\n'
          f'{_print_row(board, 7)}\n'
          f'{row_down}\n'
          f'{_print_row(board, 8)}\n'
          f'{row_up}\n'
          f'{_print_row(board, 9)}\n'
          '九    八    七    六    五    四    三    二    一    (Red)\n')

def _print_row(board, row):
    str_so_far = ''
    for i in range(0, 8):
        str_so_far += f'{_print_piece(board, row, i)}----'
    str_so_far += f'{_print_piece(board, row, 8)}'
    return str_so_far

def _print_piece(board, y, x):
    piece = board[y][x]
    try:
        return PIECES[(piece.kind, piece.is_red)]
    except AttributeError:
        if x in {0, 8}:
            return '丨'
        elif y in {0, 9}:
            return '一'
        else:
            return '十'

def _get_index_movement(board, move, is_red):
    y, x = _wxf_to_index(board, move[0:2], is_red)
    sign = move[2]
    value = int(move[3])
    piece = board[y][x]
    if sign == '.' and is_red:
        return (y, 9 - value)
    elif sign == '.' and not is_red:
        return (y, value - 1)
    elif piece.kind in {'r', 'k', 'c', 'p'}:
        if is_red and sign == '+' or not is_red and sign == '-':
            return (y - value, x)
        else:
            return (y + value, x)
    elif piece.kind == 'h' and is_red:
        vert = 3 - abs(x - (9 - value))
        if sign == '+':
            return (y - vert, 9 - value)
        else:
            return (y + vert, 9 - value)
    elif piece.kind == 'h' and not is_red:
        vert = 3 - abs(x - (value - 1))
        if sign == '+':
            return (y + vert, value - 1)
        else:
            return (y - vert, value - 1)
    else:
        if piece.kind == 'e':
            vert = 2
        else:
            vert = 1

        if sign == '+' and is_red:
            return (y - vert, 9 - value)
        elif sign == '-' and is_red:
            return (y + vert, 9 - value)
        elif sign == '+' and not is_red:
            return (y + vert, value - 1)
        else:  # sign == '-' and not is_red
            return (y - vert, value - 1)

def _get_wxf_movement(board, start, end, is_red):
    move_start = _index_to_wxf(board, start, is_red)

    if start[0] == end[0]:
        if is_red:
            return move_start + '.' + str(9 - end[1])
        else:
            return move_start + '.' + str(end[1] + 1)
    elif start[1] == end[1]:
        if (is_red and end[0] < start[0]) or (not is_red and end[0] > start[0]):
            return move_start + '+' + str(abs(end[0] - start[0]))
        else:
            return move_start + '-' + str(abs(end[0] - start[0]))
    else:
        if is_red:
            move_end = str(9 - end[1])
        else:
            move_end = str(end[1] + 1)

        if (is_red and end[0] < start[0]) or (not is_red and end[0] > start[0]):
            return move_start + '+' + move_end
        else:
            return move_start + '-' + move_end

def _wxf_to_index(board, piece, is_red):
    piece_lower = piece.lower()
    piece_type = piece_lower[0]
    if len(piece_lower) < 2:  # Kiểm tra độ dài chuỗi
        print_board(board)
        print(f"Invalid WXF move: {piece}")
        raise ValueError(f"Invalid WXF move: {piece} (too short)")
    location = piece_lower[1]

    if piece_type.isdigit():
        return _wxf_to_index_more_than_three_aligned(board, piece, is_red)
    if location in {'+', '-'}:
        return _wxf_to_index_two_aligned(board, piece, is_red)
    else:
        if is_red:
            x = 9 - int(location)
        else:
            x = int(location) - 1

        y = 0
        while y <= 9 and board[y][x] != _Piece(piece_type, is_red):
            y += 1

        if y > 9:
            print_board(board)
            print(piece)
            raise ValueError('Invalid piece')
        else:
            return (y, x)

def _wxf_to_index_two_aligned(board, piece, is_red):
    piece_lower = piece.lower()
    piece_type = piece_lower[0]
    location = piece_lower[1]

    y, x = 0, 0

    locations_so_far = []
    while y <= 9:
        if board[y][x] == _Piece(piece_type, is_red):
            locations_so_far.append((y, x))
        x += 1
        if x >= 9:
            y += 1
            x = 0

    coord1, coord2 = (), ()
    out = False

    for first_piece in locations_so_far:
        for second_piece in locations_so_far:
            if first_piece != second_piece and first_piece[1] == second_piece[1]:
                coord1, coord2 = sorted((first_piece, second_piece))
                out = True
                break
        if out:
            break

    if coord1 == () and coord2 == ():
        raise ValueError('Invalid piece')

    if (is_red and location == '+') or (not is_red and location == '-'):
        return coord1
    else:
        return coord2

def _wxf_to_index_more_than_three_aligned(board, piece, is_red):
    piece_lower = piece.lower()
    piece_type = piece_lower[0]

    y, x = 0, 0

    locations_so_far = []
    while y <= 9:
        if board[y][x] is not None and board[y][x].kind == 'p' and board[y][x].is_red == is_red:
            locations_so_far.append((y, x))
        x += 1
        if x >= 9:
            y += 1
            x = 0
    if not locations_so_far:  # Không tìm thấy quân cờ nào
        print_board(board)
        print(piece)
        raise ValueError('Invalid piece - no pawns found')

    x_counts = {}
    for y, x in locations_so_far:
        x_counts[x] = x_counts.get(x, 0) + 1

    if not x_counts:  # Trường hợp không có quân cờ nào (không nên xảy ra do check ở trên)
        print_board(board)
        print(piece)
        raise ValueError('Invalid piece - no pawns found')

    mode = max(x_counts.items(), key=lambda item: item[1])[0]
    aligned_pieces = [l for l in locations_so_far if l[1] == mode]
    aligned_pieces.sort()

    if len(aligned_pieces) < 3:
        print_board(board)
        print(piece)
        raise ValueError('Invalid piece - less than 3 aligned pawns')

    if is_red:
        return aligned_pieces[int(piece_type) - 1]
    else:
        return aligned_pieces[-int(piece_type)]

def _index_to_wxf(board, pos, is_red):
    piece = board[pos[0]][pos[1]]
    if piece is None or piece.is_red != is_red:
        raise ValueError

    piece_type = piece.kind
    x = pos[1]

    pieces = []
    y = 0
    while y <= 9:
        if board[y][x] == _Piece(piece_type, is_red):
            pieces.append((y, x))
        y += 1

    if len(pieces) == 1:
        if is_red:
            return piece_type + str(9 - x)
        else:
            return piece_type + str(x + 1)
    elif len(pieces) == 2:
        if (is_red and pieces.index((pos[0], pos[1])) == 1) \
           or (not is_red and pieces.index((pos[0], pos[1])) == 0):
            return piece_type + '-'
        else:
            return piece_type + '+'
    else:
        if is_red:
            return str(pieces.index((pos[0], pos[1])) + 1) + str(9 - x)
        else:
            return str(len(pieces) - pieces.index((pos[0], pos[1]))) + str(x + 1)

def calculate_absolute_points(board):
    points_so_far = 0
    for pos in [(y, x) for y in range(0, 10) for x in range(0, 9)]:
        piece = board[pos[0]][pos[1]]
        if piece is None:
            continue
        else:
            if piece.kind == 'p':
                points_so_far += _absolute_pawn(board, pos)
            elif piece.kind == 'h':
                points_so_far += _absolute_horse(board, pos)
            elif piece.kind == 'e':
                points_so_far += _absolute_elephant(board, pos)
            elif piece.kind == 'c':
                points_so_far += _absolute_cannon(board, pos)
            elif piece.kind == 'r':
                points_so_far += _absolute_chariot(board, pos)
            elif piece.kind == 'k':
                points_so_far += _absolute_king(board, pos)
            else:
                points_so_far += _absolute_advisor(board, pos)
    return points_so_far

def _absolute_pawn(board, pos):
    points_so_far = 0
    piece = board[pos[0]][pos[1]]
    if piece.is_red:
        side = 1
    else:
        side = -1

    if (piece.is_red and pos[0] <= 4) or (not piece.is_red and pos[0] >= 5):
        points_so_far += 200 * side
    else:
        points_so_far += 100 * side

    return points_so_far

def _absolute_horse(board, pos):
    piece = board[pos[0]][pos[1]]
    if piece.is_red:
        side = 1
    else:
        side = -1

    points_so_far = side * 400

    if pos[0] == 1 and pos[1] in {2, 6} and side == 1:
        points_so_far += 70
    elif pos[0] == 8 and pos[1] in {2, 6} and side == -1:
        points_so_far -= 70

    if pos[0] == 2 and pos[1] in {3, 5} and side == 1:
        points_so_far += 30
    elif pos[0] == 7 and pos[1] in {3, 5} and side == -1:
        points_so_far -= 30

    return points_so_far

def _absolute_elephant(board, pos):
    piece = board[pos[0]][pos[1]]
    if piece.is_red:
        side = 1
    else:
        side = -1

    points_so_far = side * 200

    if pos[0] == 7 and pos[1] == 4 and side == 1:
        points_so_far += 20
    elif pos[0] == 2 and pos[1] == 4 and side == -1:
        points_so_far -= 20

    if pos[0] == 7 and pos[1] in {0, 8} and side == 1:
        points_so_far -= 10
    elif pos[0] == 2 and pos[1] in {0, 8} and side == -1:
        points_so_far += 10

    return points_so_far

def _absolute_cannon(board, pos):
    piece = board[pos[0]][pos[1]]
    if piece.is_red:
        side = 1
    else:
        side = -1

    points_so_far = side * 450

    if pos[1] == 4 and pos[0] in {3, 4, 5} and side == 1:
        points_so_far += 60
    elif pos[1] == 4 and pos[0] in {4, 5, 6} and side == -1:
        points_so_far -= 60

    if pos[0] == 0 and pos[1] in {0, 1, 7, 8} and side == 1:
        points_so_far += 30
    elif pos[0] == 9 and pos[1] in {0, 1, 7, 8} and side == -1:
        points_so_far -= 30

    return points_so_far

def _absolute_king(board, pos):
    piece = board[pos[0]][pos[1]]
    if piece.is_red:
        side = 1
    else:
        side = -1

    points_so_far = side * 10000

    if pos[0] in {7, 8} and side == 1:
        points_so_far -= 10
    elif pos[0] in {1, 2} and side == -1:
        points_so_far += 10

    game = ChessGame(board, red_active=piece.is_red)

    if game.is_in_check(board, piece.is_red):
        points_so_far -= 500 * side
    return points_so_far

def _absolute_chariot(board, pos):
    piece = board[pos[0]][pos[1]]
    if piece.is_red:
        side = 1
    else:
        side = -1

    points_so_far = side * 900

    if pos[0] == 9 and pos[1] in {0, 8} and side == 1:
        points_so_far -= 10
    elif pos[0] == 0 and pos[1] in {0, 8} and side == -1:
        points_so_far += 10

    return points_so_far

def _absolute_advisor(board, pos):
    piece = board[pos[0]][pos[1]]
    if piece.is_red:
        side = 1
    else:
        side = -1

    points_so_far = side * 200

    if pos[0] == 7 and pos[1] in {3, 5} and side == 1:
        points_so_far -= 10
    elif pos[0] == 2 and pos[1] in {3, 5} and side == -1:
        points_so_far += 10

    return points_so_far

def piece_count(board):
    pieces_so_far = 0
    for pos in [(y, x) for y in range(0, 10) for x in range(0, 9)]:
        if board[pos[0]][pos[1]] is not None:
            pieces_so_far += 1

    return pieces_so_far

class _Piece:
    def __init__(self, kind, is_red):
        self.kind = kind
        self.is_red = is_red

    def __eq__(self, other):
        if other is None:
            return False
        return self.kind == other.kind and self.is_red == other.is_red

    def __hash__(self):
        return hash((self.kind, self.is_red))

if __name__ == '__main__':
    import python_ta.contracts
    python_ta.contracts.check_all_contracts()
    import python_ta
    python_ta.check_all(config={
        'max-line-length': 100,
        'disable': ['E1136', 'E9989', 'E9998', 'W1401', 'R0201', 'R1702', 'R0912', 'R0913'],
        'extra-imports': ['typing', 'statistics', 'copy']
    })
    import doctest
    doctest.testmod()