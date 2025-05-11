import tkinter as tk
from tkinter import ttk, messagebox
from Game_UI import Game
from AlphaBetaAgent import AlphaBetaAgent
from MinimaxAgent import MinimaxAgent
from PIL import Image, ImageTk
import os
import logging

# Cấu hình logging để ghi vào cả hai file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('minimax.log', mode='a', encoding='utf-8'),
        logging.FileHandler('alphabeta.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class ChineseChessAppWithAgents:
    def __init__(self, root):
        self.root = root
        self.root.title("Chinese Chess")
        self.root.geometry("600x500")
        
        # Sound states
        self.bgm_enabled = True
        self.sfx_enabled = True
        
        # Agent selection
        self.selected_agent = tk.StringVar(value="Alpha-Beta")  # Mặc định là Alpha-Beta
        self.max_depth = 3  # Độ sâu mặc định
        
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
        
        # Agent selection
        agent_frame = ttk.LabelFrame(main_frame, text="Select Opponent", padding=10)
        agent_frame.pack(pady=10, fill=tk.X)
        
        ttk.Radiobutton(agent_frame, text="Alpha-Beta Agent", value="Alpha-Beta", variable=self.selected_agent).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(agent_frame, text="Minimax Agent", value="Minimax", variable=self.selected_agent).pack(side=tk.LEFT, padx=10)
        
        # Depth selection
        depth_frame = ttk.LabelFrame(main_frame, text="Select Depth", padding=10)
        depth_frame.pack(pady=10, fill=tk.X)
        
        self.depth_var = tk.StringVar(value=str(self.max_depth))
        ttk.Label(depth_frame, text="Depth:").pack(side=tk.LEFT)
        depth_entry = ttk.Entry(depth_frame, textvariable=self.depth_var, width=5)
        depth_entry.pack(side=tk.LEFT, padx=5)
        
        # Sound settings
        settings_frame = ttk.LabelFrame(main_frame, text="Sound Settings", padding=10)
        settings_frame.pack(pady=10, fill=tk.X)
        
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
        
        # Start button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        play_btn = ttk.Button(button_frame, text="Start Game", command=self.start_game)
        play_btn.pack(side=tk.LEFT, padx=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to play")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(20, 0))
    
    def toggle_bgm(self):
        self.bgm_enabled = not self.bgm_enabled
        self.bgm_button.config(image=self.bgm_icon if self.bgm_enabled else self.bgm_muted_icon)
    
    def toggle_sfx(self):
        self.sfx_enabled = not self.sfx_enabled
        self.sfx_button.config(image=self.sfx_icon if self.sfx_enabled else self.sfx_muted_icon)
    
    def start_game(self):
        try:
            self.max_depth = int(self.depth_var.get())
            if self.max_depth < 1:
                raise ValueError("Depth must be at least 1")
        except ValueError as e:
            messagebox.showerror("Invalid Depth", f"Depth must be a positive integer: {str(e)}")
            self.status_var.set("Invalid depth")
            return

        agent_type = self.selected_agent.get()
        self.status_var.set(f"Starting game with {agent_type} (depth {self.max_depth})...")
        self.root.after(100, self._run_game)
    
    def _run_game(self):
        try:
            agent_type = self.selected_agent.get()
            if agent_type == "Alpha-Beta":
                player = AlphaBetaAgent(max_depth=self.max_depth)
            else:  # Minimax
                player = MinimaxAgent(max_depth=self.max_depth)

            game = Game(player)
            game.run()
            self.status_var.set("Game ended")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start game: {str(e)}")
            self.status_var.set("Error starting game")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChineseChessAppWithAgents(root)
    root.mainloop()