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
    def __init__(self, board = None, red_active = True, move_count = 0):
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

        self._recalculate_valid_moves() 

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

        self._is_red_active = not self._is_red_active  
        self._move_count += 1

        self._recalculate_valid_moves()

    def copy_and_make_move(self, move):
        if move not in self._valid_moves:
            raise ValueError(f'Move "{move}" is not valid')

        return ChessGame(board=self._board_after_move(move, self._is_red_active), red_active=not self._is_red_active, move_count=self._move_count + 1)

    def is_red_move(self):
        return self._is_red_active

    def get_winner(self):
        if self._move_count >= _MAX_MOVES: 
            return 'Draw'
        elif all(self._board[y][x] != _Piece('k', True)
                 for y in range(0, 10) for x in range(0, 9)):  
            return 'Black'
        elif all(self._board[y][x] != _Piece('k', False)
                 for y in range(0, 10) for x in range(0, 9)): 
            return 'Red'
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

    def _find_moves_in_direction(self, board, pos, is_red, direction, limit = None, capture = None):
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

    def _recalculate_valid_moves(self):
        self._valid_moves = self._calculate_moves_for_board(self._board, self._is_red_active)


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
        if board[y][x] == _Piece('p', is_red):
            locations_so_far.append((y, x))
        x += 1
        if x >= 9:
            y += 1
            x = 0

    x_values = [l[1] for l in locations_so_far]
    mode = statistics.mode(x_values)
    aligned_pieces = [l for l in locations_so_far if l[1] == mode]
    aligned_pieces.sort()

    if len(aligned_pieces) < 3:
        raise ValueError('Invalid piece')

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


if __name__ == '__main__':
    # import python_ta.contracts
    # python_ta.contracts.check_all_contracts()

    # import python_ta
    # python_ta.check_all(config={
    #     'max-line-length': 100,
    #     # Note: we ran PyTA against the starter file a2_minichess.py given in Assignment 2,
    #     # and disabled the ones that file did not pass either.
    #     'disable': ['E1136', 'E9989', 'E9998', 'W1401', 'R0201', 'R1702', 'R0912', 'R0913'],
    #     'extra-imports': ['typing', 'statistics', 'copy']
    # })

    import doctest
    doctest.testmod()