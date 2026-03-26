import tkinter as tk
from tkinter import messagebox
import threading

from game.othello import Othello
from game.utils import Player
from agent.agent import Agent

CELL_SIZE = 60
BOARD_SIZE = 8
PADDING = 10
AI_DELAY_MS = 600


class OthelloGUI:
    def __init__(self, root):
        self.root = root
        root.title("Othello - GUI")

        self.mode = tk.IntVar(value=2)

        controls = tk.Frame(root)
        controls.pack(padx=8, pady=8, anchor='w')

        tk.Label(controls, text="Modo:").pack(side='left')
        tk.Radiobutton(controls, text="Pessoa x Pessoa", variable=self.mode, value=1).pack(side='left')
        tk.Radiobutton(controls, text="Pessoa(preto) x Agente", variable=self.mode, value=2).pack(side='left')
        tk.Radiobutton(controls, text="Agente x Agente", variable=self.mode, value=3).pack(side='left')

        tk.Button(controls, text="Novo Jogo", command=self.start_new_game).pack(side='left', padx=8)

        status_frame = tk.Frame(root)
        status_frame.pack(padx=8, pady=(0, 8), fill='x')

        self.score_label = tk.Label(status_frame, text="BLACK: 0  WHITE: 0")
        self.score_label.pack(side='left')

        self.status_label = tk.Label(status_frame, text="Status: Pronto")
        self.status_label.pack(side='right')

        canvas_w = BOARD_SIZE * CELL_SIZE + 2 * PADDING
        canvas_h = BOARD_SIZE * CELL_SIZE + 2 * PADDING

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg='#228B22')
        self.canvas.pack(padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # estado interno
        self.black_is_computer = False
        self.white_is_computer = False
        self.possible_moves = {}
        self.consecutive_passes = 0
        self.ai_pending = False

        self.draw_grid()
        self.start_new_game()

    def start_new_game(self):
        mode = self.mode.get()

        if mode == 1:
            self.black_is_computer = False
            self.white_is_computer = False
        elif mode == 2:
            self.black_is_computer = False
            self.white_is_computer = True
        else:
            self.black_is_computer = True
            self.white_is_computer = True

        Othello.setInitialParameters()
        self.consecutive_passes = 0

        self.update_board()
        self.status_label.config(text=f"Turn: {Othello.turn.name}")

        if self.current_player_is_ai():
            self.root.after(AI_DELAY_MS, self.perform_ai_move)

    def draw_grid(self):
        self.canvas.delete("grid")

        for i in range(BOARD_SIZE + 1):
            y = PADDING + i * CELL_SIZE
            self.canvas.create_line(PADDING, y, PADDING + BOARD_SIZE * CELL_SIZE, y, tags="grid")

        for j in range(BOARD_SIZE + 1):
            x = PADDING + j * CELL_SIZE
            self.canvas.create_line(x, PADDING, x, PADDING + BOARD_SIZE * CELL_SIZE, tags="grid")

    def update_board(self):
        self.canvas.delete("disc")
        self.canvas.delete("hint")

        board = Othello.board

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                x = PADDING + j * CELL_SIZE + CELL_SIZE // 2
                y = PADDING + i * CELL_SIZE + CELL_SIZE // 2

                if board[i][j] == Player.BLACK:
                    self.canvas.create_oval(x-22, y-22, x+22, y+22, fill='black', tags="disc")
                elif board[i][j] == Player.WHITE:
                    self.canvas.create_oval(x-22, y-22, x+22, y+22, fill='white', tags="disc")

        plays = Othello.possiblePlays()
        self.possible_moves = plays.playsList

        for (r, c) in self.possible_moves.keys():
            x = PADDING + c * CELL_SIZE + CELL_SIZE // 2
            y = PADDING + r * CELL_SIZE + CELL_SIZE // 2
            self.canvas.create_oval(x-6, y-6, x+6, y+6, fill='yellow', tags="hint")

        self.score_label.config(
            text=f"BLACK: {Othello.score['BLACK']}  WHITE: {Othello.score['WHITE']}"
        )
        self.status_label.config(text=f"Turn: {Othello.turn.name}")

    def canvas_pos_to_cell(self, x, y):
        if x < PADDING or y < PADDING:
            return None

        col = (x - PADDING) // CELL_SIZE
        row = (y - PADDING) // CELL_SIZE

        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return (row, col)

        return None

    def on_canvas_click(self, event):
        if self.current_player_is_ai():
            return

        cell = self.canvas_pos_to_cell(event.x, event.y)

        if not cell or cell not in self.possible_moves:
            return

        directions = self.possible_moves[cell]
        Othello.propagateChoose(cell, directions)

        self.consecutive_passes = 0
        self.after_move_cleanup()

    def current_player_is_ai(self):
        return (
            (Othello.turn == Player.BLACK and self.black_is_computer) or
            (Othello.turn == Player.WHITE and self.white_is_computer)
        )

    # ---------------- THREADING ---------------- #

    def perform_ai_move(self):
        if not self.current_player_is_ai() or self.ai_pending:
            return

        self.ai_pending = True
        self.status_label.config(text=f"{Othello.turn.name} thinking...")

        thread = threading.Thread(target=self.ai_worker, daemon=True)
        thread.start()

    def ai_worker(self):
        plays = Othello.possiblePlays()

        if not plays.hasPossiblePlays:
            self.root.after(0, self.handle_ai_pass)
            return

        agent = Agent(Othello.turn, Othello.opponent, Othello.board)

        try:
            chosen = agent.choosePlay()
        except Exception:
            chosen = list(plays.playsList.keys())[0]

        if chosen not in plays.playsList:
            if isinstance(chosen, int):
                keys = list(plays.playsList.keys())
                chosen = keys[chosen % len(keys)]

        self.root.after(0, self.apply_ai_move, chosen, plays.playsList[chosen])

    def apply_ai_move(self, chosen, directions):
        self.ai_pending = False

        Othello.propagateChoose(chosen, directions)
        self.consecutive_passes = 0

        self.after_move_cleanup()

        if self.current_player_is_ai():
            self.root.after(AI_DELAY_MS, self.perform_ai_move)

    def handle_ai_pass(self):
        self.ai_pending = False

        self.consecutive_passes += 1

        if self.consecutive_passes >= 2:
            Othello.endGameByScore()
            self.finish_game()
            return

        Othello.changeTurn()
        self.update_board()

        if self.current_player_is_ai():
            self.root.after(AI_DELAY_MS, self.perform_ai_move)

    # ------------------------------------------ #

    def after_move_cleanup(self):
        self.update_board()
        Othello.verifyWinner()

        if Othello.hasWinner:
            self.finish_game()
            return

        plays = Othello.possiblePlays()

        if not plays.hasPossiblePlays:
            self.consecutive_passes += 1

            if self.consecutive_passes >= 2:
                Othello.endGameByScore()
                self.finish_game()
                return

            messagebox.showinfo("Pass", f"Player {Othello.turn.name} passes.")
            Othello.changeTurn()
            self.update_board()

            if self.current_player_is_ai():
                self.root.after(AI_DELAY_MS, self.perform_ai_move)

            return

        else:
            self.consecutive_passes = 0

        Othello.changeTurn()
        self.update_board()

        if self.current_player_is_ai():
            self.root.after(AI_DELAY_MS, self.perform_ai_move)

    def finish_game(self):
        self.update_board()

        if Othello.winner != Player.EMPTY:
            messagebox.showinfo(
                "Game Over",
                f"{Othello.winner.name} wins! "
                f"{Othello.score[Othello.winner.name]}:"
                f"{Othello.score[Othello.loser.name]}"
            )
        else:
            messagebox.showinfo(
                "Game Over",
                f"Tie! {Othello.score['BLACK']}:{Othello.score['WHITE']}"
            )


def run():
    root = tk.Tk()
    app = OthelloGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run()