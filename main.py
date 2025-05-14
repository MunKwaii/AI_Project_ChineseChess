import pygame
from pygame.locals import *
from Game_UI import Game
from Player import ChessAgent
from AlphaBetaAgent import AlphaBetaAgent
import os

class ChineseChessApp:
    def __init__(self):
        pygame.init()
        self.screen_width = 900
        self.screen_height = 700
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Chinese Chess")
<<<<<<< HEAD

        self.chess_agent = ChessAgent(state_size=(10, 9), action_size=4000, use_api=False)
        self.alpha_beta_agent = AlphaBetaAgent(max_depth=2, xml_file='tree.xml')
=======

        self.chess_agent = ChessAgent(state_size=(10, 9), action_size=4000)
        model_path = "trained_models/chinese_chess_alpha.pth"
        if os.path.exists(model_path):
            print(f"Loaded ChessAgent model from {model_path}")
        self.alpha_beta_agent = AlphaBetaAgent(max_depth=3)

>>>>>>> f17bfe4321147cbf284676e016be50f7c215acf3
        self.BACKGROUND_COLOR = (231, 191, 118)  
        self.BUTTON_COLOR = (231, 191, 118)  
        self.BUTTON_HOVER_COLOR = (139, 69, 19)  
        self.TEXT_COLOR = (0, 0, 0)  
        self.TITLE_COLOR = (231, 191, 118) 
        self.TITLE_SHADOW_COLOR = (0, 0, 0)  
        self.BUTTON_BORDER_COLOR = (0, 0, 0)  

        self.title_font = pygame.font.SysFont('Times New Roman', 80, bold=True)
        self.button_font = pygame.font.SysFont('Arial', 32, bold=True)

        self.chess_agent_button_rect = pygame.Rect(250, 250, 430, 80)
        self.alpha_beta_button_rect = pygame.Rect(250, 380, 430, 80)

        try:
            self.background = pygame.image.load('Effect_Graphics/Background.jpg')
            self.background = pygame.transform.scale(self.background, (self.screen_width, self.screen_height))
        except pygame.error:
            self.background = None

    def draw_button(self, rect, text, hover=False):
        color = self.BUTTON_HOVER_COLOR if hover else self.BUTTON_COLOR
        pygame.draw.rect(self.screen, self.BUTTON_BORDER_COLOR, rect, border_radius=12)
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, color, inner_rect, border_radius=10)
        text_surface = self.button_font.render(text, True, self.TEXT_COLOR)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def draw(self):
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill(self.BACKGROUND_COLOR)

        title_surface = self.title_font.render("CHINESE CHESS", True, self.TITLE_COLOR)
        shadow_surface = self.title_font.render("CHINESE CHESS", True, self.TITLE_SHADOW_COLOR)
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, 100))
        shadow_rect = title_rect.move(4, 4)
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(title_surface, title_rect)

        mouse_pos = pygame.mouse.get_pos()
        chess_agent_hover = self.chess_agent_button_rect.collidepoint(mouse_pos)
        alpha_beta_hover = self.alpha_beta_button_rect.collidepoint(mouse_pos)

        self.draw_button(self.chess_agent_button_rect, "MCTS With Policy/Value Network", chess_agent_hover)
        self.draw_button(self.alpha_beta_button_rect, "Minimax With Alpha-Beta Pruning", alpha_beta_hover)

        pygame.display.flip()

    def start_game(self, agent):
        try:
            game = Game(agent)
            game.run()
        except Exception as e:
            print(f"Error starting game: {str(e)}")  

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == MOUSEBUTTONDOWN:
                    if self.chess_agent_button_rect.collidepoint(event.pos):
                        self.start_game(self.chess_agent)
                    elif self.alpha_beta_button_rect.collidepoint(event.pos):
                        self.start_game(self.alpha_beta_agent)

            self.draw()

        pygame.quit()

if __name__ == '__main__':
    app = ChineseChessApp()
    app.run()