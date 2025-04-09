import tkinter as tk
from tkinter import ttk, messagebox
from Game_UI import Game  # Giả định đây là file Game.py bạn đã có
from Player import MCTSPlayer, DQNAgent  # Dùng MCTSPlayer của bạn
from PIL import Image, ImageTk
import os
import threading

class ChineseChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chinese Chess")
        self.root.geometry("600x500")
        
        # Load MCTS player
        self.loaded_agent = DQNAgent.load_full_model("trained_models/chinese_chess_dqn")
        self.player = MCTSPlayer(iterations=100, simulate_fn=self.loaded_agent)
        
        # Sound states
        self.bgm_enabled = True
        self.sfx_enabled = True
        
        try:
            self.setup_ui()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize UI: {str(e)}")
            self.root.destroy()

    def load_images(self):
        """Load speaker icons from files"""
        try:
            self.bgm_icon = ImageTk.PhotoImage(Image.open("Effect_Graphics/Icon/speaker.webp").resize((24, 24)))
            self.bgm_muted_icon = ImageTk.PhotoImage(Image.open("Effect_Graphics/Icon/speaker_muted.png").resize((24, 24)))
            self.sfx_icon = ImageTk.PhotoImage(Image.open("Effect_Graphics/Icon/speaker.webp").resize((24, 24)))
            self.sfx_muted_icon = ImageTk.PhotoImage(Image.open("Effect_Graphics/Icon/speaker_muted.png").resize((24, 24)))
        except FileNotFoundError:
            from tkinter import font as tkfont
            self.bgm_icon = self.create_text_icon("🔊")
            self.bgm_muted_icon = self.create_text_icon("🔇")
            self.sfx_icon = self.create_text_icon("🔊")
            self.sfx_muted_icon = self.create_text_icon("🔇")

    def create_text_icon(self, text):
        """Create a simple text-based icon as fallback"""
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.text((16, 16), text, font=font, anchor="mm", fill="black")
        return ImageTk.PhotoImage(image)

    def setup_ui(self):
        self.load_images()

        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        title_label = ttk.Label(main_frame, text="Chinese Chess", font=("Arial", 24, "bold"))
        title_label.pack(pady=10)
        
        settings_frame = ttk.LabelFrame(main_frame, text="Sound Settings", padding=10)
        settings_frame.pack(pady=20, fill=tk.X)
        
        sound_frame = ttk.Frame(settings_frame)
        sound_frame.pack(pady=5)
        
        self.bgm_button = ttk.Button(
            sound_frame,
            image=self.bgm_icon if self.bgm_enabled else self.bgm_muted_icon,
            command=self.toggle_bgm
        )
        self.bgm_button.pack(side=tk.LEFT, padx=10)
        ttk.Label(sound_frame, text="Background Music").pack(side=tk.LEFT)
        
        self.sfx_button = ttk.Button(
            sound_frame,
            image=self.sfx_icon if self.sfx_enabled else self.sfx_muted_icon,
            command=self.toggle_sfx
        )
        self.sfx_button.pack(side=tk.LEFT, padx=30)
        ttk.Label(sound_frame, text="Sound Effects").pack(side=tk.LEFT)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        play_btn = ttk.Button(button_frame, text="Start Game", command=self.start_game)
        play_btn.pack(side=tk.LEFT, padx=10)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready with MCTS Player")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(20, 0))
    
    def toggle_bgm(self):
        self.bgm_enabled = not self.bgm_enabled
        self.bgm_button.config(image=self.bgm_icon if self.bgm_enabled else self.bgm_muted_icon)
    
    def toggle_sfx(self):
        self.sfx_enabled = not self.sfx_enabled
        self.sfx_button.config(image=self.sfx_icon if self.sfx_enabled else self.sfx_muted_icon)
    
    def start_game(self):
        self.status_var.set("Starting game with MCTS...")
        self.root.after(100, self._run_game)
    
    def _run_game(self):
        try:
            game = Game(self.player, self.bgm_enabled, self.sfx_enabled)
            game.run()
            self.status_var.set("Game ended")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start game: {str(e)}")
            self.status_var.set("Error starting game")

if __name__ == '__main__':
    root = tk.Tk()
    app = ChineseChessApp(root)
    root.mainloop()