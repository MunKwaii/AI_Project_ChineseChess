import pygame
from Chinese_Chess_Game_Rules import ChessGame
import Chinese_Chess_Game_Rules
from Chinese_Chess_Game_Rules import ChessGame, piece_count

class Game:
    def __init__(self, player):
        pygame.init()

        self.board_width = 720
        self.board_height = 800
        self.cell_size = 80

        self.opponent = player
        self._screen = pygame.display.set_mode((self.board_width, self.board_height))
        self._game = ChessGame()
        self._curr_coord = ()
        self._ready_to_move = False
        self._movement_indices = []
        self._game_ended = False

        pygame.display.set_caption('Chinese Chess!')
        self._change_icon()

        global IMAGE_DICT
        # global SOUND_DICT
        global COLOR_DICT
        global FONT_DICT
        IMAGE_DICT = self._load_images()
        # SOUND_DICT = self._load_sound()
        COLOR_DICT = self._define_color()
        FONT_DICT = self._define_font()

        self._screen.fill(COLOR_DICT['background_color'])
        # self.display_instructions()

    def _load_images(self):
        board_image = pygame.image.load('Effect_Graphics/Board/board_plain.webp')
        # board_image = pygame.transform.scale(board_image, (self.board_width, self.board_height))

        piece_size = (57, 57)  # Kích thước mới cho quân cờ
        
        black_advisor = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/black_advisor.webp'), piece_size)
        black_elephant = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/black_bishop.webp'), piece_size)
        black_cannon = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/black_cannon.webp'), piece_size)
        black_king = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/black_king.webp'), piece_size)
        black_horse = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/black_knight.webp'), piece_size)
        black_pawn = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/black_pawn.webp'), piece_size)
        black_chariot = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/black_rook.webp'), piece_size)
        red_advisor = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/red_advisor.webp'), piece_size)
        red_elephant = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/red_bishop.webp'), piece_size)
        red_cannon = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/red_cannon.webp'), piece_size)
        red_king = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/red_king.webp'), piece_size)
        red_horse = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/red_knight.webp'), piece_size)
        red_pawn = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/red_pawn.webp'), piece_size)
        red_chariot = pygame.transform.scale(pygame.image.load('Effect_Graphics/Piece/red_rook.webp'), piece_size)
        
        possible_move_frame = pygame.transform.scale(pygame.image.load('Effect_Graphics/path_go.webp'), (60, 60))
        selected_frame = pygame.transform.scale(pygame.image.load('Effect_Graphics/selection.webp'), (50, 50))

        return {('r', False): black_chariot, ('h', False): black_horse,
                ('e', False): black_elephant, ('a', False): black_advisor,
                ('k', False): black_king, ('c', False): black_cannon,
                ('p', False): black_pawn, ('r', True): red_chariot,
                ('h', True): red_horse, ('e', True): red_elephant,
                ('a', True): red_advisor, ('k', True): red_king,
                ('c', True): red_cannon, ('p', True): red_pawn,
                'board_image': board_image, 'possible_move_frame': possible_move_frame, 'selected_frame': selected_frame}
        
    def _define_color(self):
        black = (0, 0, 0)
        red = (255, 0, 0)
        blue = (0, 0, 255)
        white = (255, 255, 255)
        background_color = (231, 191, 118)

        return {'black': black,
                'red': red,
                'blue': blue,
                'white': white,
                'background_color': background_color}

    def _define_font(self) -> dict:
        font_bold = pygame.font.SysFont('American Typewriter', 36, bold=True)
        font = pygame.font.SysFont('American Typewriter', 24)
        text = pygame.font.SysFont('Arial', 24)

        return {'font_bold': font_bold, 'font': font, 'text': text}

    def _change_icon(self):
        red_king = pygame.image.load('Effect_Graphics/Piece/red_king.webp')
        icon = pygame.Surface(red_king.get_size())
        icon.blit(red_king, (0, 0))
        pygame.display.set_icon(icon)

    def run(self):
        self._print_game()
        red_status_rect = IMAGE_DICT['possible_move_frame'].get_rect(center=(900, 900))
        black_status_rect = IMAGE_DICT['possible_move_frame'].get_rect(center=(900, 900))
        self._screen.blit(IMAGE_DICT['possible_move_frame'], red_status_rect)
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print('Thanks for playing!')
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN and not self._ready_to_move and not self._game_ended:
                    self._get_possible_moves_for_piece(event.pos)
                    pygame.display.flip()
                elif event.type == pygame.MOUSEBUTTONDOWN and self._ready_to_move and not self._game_ended:
                    new_coordinate = pixel_to_coordinate((event.pos[0], event.pos[1]))
                    if new_coordinate not in self._movement_indices: 
                        self._screen.fill(COLOR_DICT['background_color'])
                        self._print_game()
                        self._ready_to_move = False
                        pygame.display.flip()
                    else:  
                        try:
                            wxf_move = Chinese_Chess_Game_Rules._get_wxf_movement(self._game.get_board(), self._curr_coord, new_coordinate, True)
                        except ValueError:  
                            self._screen.fill(COLOR_DICT['background_color'])
                            self._print_game()
                            self._ready_to_move = False
                            pygame.display.flip()
                            continue
                        self._make_a_move(wxf_move, red_status_rect, black_status_rect, True)
                        if self._check_for_end(): 
                            continue

                        opponent_wxf_move = self.opponent.get_move(self._game)
<<<<<<< HEAD
                        # opponent_wxf_move = self.opponent.get_move(self._game, simulations=200, c_puct=2.0, policy_threshold=0.7)
=======
>>>>>>> f17bfe4321147cbf284676e016be50f7c215acf3
                        self._curr_coord = Chinese_Chess_Game_Rules._wxf_to_index(self._game.get_board(), opponent_wxf_move, False)
                        self._make_a_move(opponent_wxf_move, black_status_rect, red_status_rect, False)
                        if self._check_for_end():  
                            continue

    def _make_a_move(self, wxf_move, old_status_rect, new_status_rect, is_red):
        destination = Chinese_Chess_Game_Rules._get_index_movement(self._game.get_board(), wxf_move, is_red)
        pieces_before = piece_count(self._game.get_board())
        self._game.make_move(wxf_move)
        pieces_after = piece_count(self._game.get_board())
        
        self._screen.fill(COLOR_DICT['background_color'])
        self._print_game()

        piece_frame_coord = coordinate_to_pixel(self._curr_coord)
        piece_frame_rect = IMAGE_DICT['selected_frame'].get_rect(center=piece_frame_coord)
        self._screen.blit(IMAGE_DICT['selected_frame'], piece_frame_rect)

        piece_frame_coord_after = coordinate_to_pixel(destination)
        piece_frame_after_rect = IMAGE_DICT['selected_frame'].get_rect(center=piece_frame_coord_after)
        self._screen.blit(IMAGE_DICT['selected_frame'], piece_frame_after_rect)

        status_clear = pygame.Surface(IMAGE_DICT['possible_move_frame'].get_size())
        status_clear.fill((231, 191, 118))
        self._screen.blit(status_clear, old_status_rect)
        self._screen.blit(IMAGE_DICT['possible_move_frame'], new_status_rect)
        pygame.display.flip()

    def _print_game(self):
        self._screen.blit(IMAGE_DICT['board_image'], (0, 0)) 
        # self._screen.blit(IMAGE_DICT['coord_image'], (0, 0))  
        for pos in [(y, x) for y in range(0, 10) for x in range(0, 9)]:  
            piece = self._game.get_board()[pos[0]][pos[1]]
            if piece is not None:
                piece_coord = coordinate_to_pixel((pos[0], pos[1]))
                piece_rect = IMAGE_DICT[(piece.kind, piece.is_red)].get_rect(center=piece_coord)
                self._screen.blit(IMAGE_DICT[(piece.kind, piece.is_red)], piece_rect)

    def _get_possible_moves_for_piece(self, pos):
        possible_moves = self._game.get_valid_moves()
        self._curr_coord = pixel_to_coordinate((pos[0], pos[1]))
        if not (0 <= self._curr_coord[0] <= 9 and 0 <= self._curr_coord[1] <= 8):
            return  
        try: 
            piece_wxf = Chinese_Chess_Game_Rules._index_to_wxf(self._game.get_board(), self._curr_coord, True)
        except ValueError:
            return
        piece_frame_coord = coordinate_to_pixel(self._curr_coord)
        piece_frame_rect = IMAGE_DICT['selected_frame'].get_rect(center=piece_frame_coord)
        self._screen.blit(IMAGE_DICT['selected_frame'], piece_frame_rect)
        piece_possible_moves = [move for move in possible_moves if move[0:2] == piece_wxf]
        self._movement_indices = [Chinese_Chess_Game_Rules._get_index_movement(self._game.get_board(), move, True) for move in piece_possible_moves]
        for coord in self._movement_indices:
            frame_coord = coordinate_to_pixel(coord)
            frame_rect = IMAGE_DICT['possible_move_frame'].get_rect(center=frame_coord)
            self._screen.blit(IMAGE_DICT['possible_move_frame'], frame_rect)

        self._ready_to_move = True

    def _check_for_end(self):
        if self._game.get_winner() is not None:
            self._game_ended = True
            self._print_result(self._game.get_winner())
        else:
            pass

        return self._game_ended

    def _print_result(self, winner):
        text_dict = {'Red': 'Congratulations! You won!', 'Black': 'Too bad. You lost.', 'Draw': 'No one won.'}
        color_dict = {'Red': 'red', 'Black': 'black', 'Draw': 'blue'}
        text = text_dict[winner]
        color = color_dict[winner]
        message = FONT_DICT['font_bold'].render(text, True, color)
        message_rect = message.get_rect(center=(360, 400))
        message_surface = pygame.Surface(message.get_size())
        # message_surface.fill(COLOR_DICT['white'])
        message_surface.set_alpha(200)  
        self._screen.blit(message_surface, message_rect)
        self._screen.blit(message, message_rect)
        closing_message = FONT_DICT['font'].render('Please close this window.', True, color)
        closing_message_rect = closing_message.get_rect(center=(360, 400))
        closing_message_surface = pygame.Surface(closing_message.get_size())  
        closing_message_surface.fill(COLOR_DICT['white'])
        closing_message_surface.set_alpha(200)  
        self._screen.blit(closing_message_surface, closing_message_rect)
        self._screen.blit(closing_message, closing_message_rect)

        pygame.display.flip()  


def coordinate_to_pixel(coordinate):
    margin_x = 40.4  
    margin_y = 39.7  
    
    x = margin_x + coordinate[1] * 80.4  
    y = margin_y + coordinate[0] * 80.4
    return x, y


def pixel_to_coordinate(pixel):
    x, y = pixel
    margin_x = 40.4
    margin_y = 39.7
    
    col = round((x - margin_x) / 80.4)
    row = round((y - margin_y) / 80.4)
    
    col = max(0, min(8, col))
    row = max(0, min(9, row))
    
    return row, col