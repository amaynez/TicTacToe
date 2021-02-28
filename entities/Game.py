import numpy as np
import random


class Game:
    def __init__(self):
        self.state = np.zeros((3, 3), np.int8)
        self.PLAY_SPRITES = []
        self.score = [0, 0]
        self.winner = 0
        self.turn = 1

    def new_game(self):
        self.state = np.zeros((3, 3), np.int8)
        for player in self.PLAY_SPRITES:
            player.kill()
        self.PLAY_SPRITES = []
        self.winner = 0

    def new_play(self, row, col):
        player = self.turn
        if self.state[row, col] == 0:
            self.state[row, col] = self.turn
            self.check_winner()
            self.turn = 2 if self.turn == 1 else 1
            if self.winner > 0:
                return -1, (row, col, player)
        return 0, (row, col, player)

    def check_winner(self):
        for row_col in range(3):
            for player in range(1, 3, 1):
                if np.all(self.state[row_col, :] == player) or np.all(self.state[:, row_col] == player):
                    self.winner = player
        for player in range(1, 3, 1):
            if np.all(self.state.diagonal() == player) or np.all(np.flipud(self.state).diagonal() == player):
                self.winner = player
        if self.winner == 0:
            empty_cells = np.where(self.state == 0)
            if len(empty_cells[0]) == 0:
                self.winner = 3

    def AI_play(self):
        adversary = 1 if self.turn == 2 else 2

        # finish immediate threat from us (win)
        # and block immediate threats from adversary (don't loose)
        for player in range(3 - adversary, 3 if adversary == 2 else 0, 1 if adversary == 2 else -1):
            for row_col in range(3):
                for col_row in range(3):
                    # check rows
                    if self.state[row_col, col_row] == 0 and \
                            np.all(self.state[row_col, (np.arange(3) != col_row)] == player):
                        return self.new_play(row_col, col_row)
                    # check cols
                    if self.state[col_row, row_col] == 0 and \
                            np.all(self.state[(np.arange(3) != col_row), row_col] == player):
                        return self.new_play(col_row, row_col)
                # check main diagonal
                diagonal = np.diagonal(self.state)
                if self.state[row_col, row_col] == 0 and \
                        np.all(diagonal[(np.arange(3) != row_col)] == player):
                    return self.new_play(row_col, row_col)
                # check secondary diagonal
                diagonal = np.diagonal(np.flipud(self.state))
                if self.state[2 - row_col, row_col] == 0 and \
                        np.all(diagonal[(np.arange(3) != row_col)] == player):
                    return self.new_play( 2 - row_col, row_col)

        # else check for corner threats
        for i in range(2):
            if self.state[0, i * 2] == adversary and \
                    self.state[2, 2 - (i * 2)] == adversary and \
                    np.all(self.state[0, (1 - i, 2 - i)] == 0) and \
                    np.all(self.state[(i, 1 + i), 2] == 0) and \
                    np.all(self.state[(1 - i, 2 - i), 0] == 0) and \
                    np.all(self.state[2, (i, 1 + i)]) == 0:
                return self.new_play(0, 1)

        # else check for center corner threat
        if self.state[0, 2] == 0 and \
                self.state[1, 1] == adversary and \
                self.state[2, 2] == adversary:
            return self.new_play(0, 2)

        # else go for possible simultaneous threat from us
        # and block possible simultaneous threats from adversary
        for player in range(3 - adversary, 3 if adversary == 2 else 0, 1 if adversary == 2 else -1):
            for row_col in range(3):
                for col_row in range(3):
                    if self.state[row_col, col_row] == player and \
                            np.all(self.state[row_col, (np.arange(3) != col_row)] == 0):
                        p = [p for p in range(3) if p != col_row]
                        for i in p:
                            q = [q for q in range(3) if q != row_col]
                            for j in q:
                                if self.state[j, i] == player and \
                                        np.all(self.state[(np.arange(3) != j), i] == 0):
                                    return self.new_play(row_col, i)

        # else play center
        if self.state[1, 1] == 0:
            return self.new_play(1, 1)

        # else play corner
        for y in range(0, 3, 2):
            for x in range(0, 3, 2):
                if self.state[y, x] == 0:
                    return self.new_play(y, x)

        # else play random move from the available cells
        empty_cells = np.where(self.state == 0)
        choice = random.choice(range(len(empty_cells[0])))
        x = empty_cells[1][choice]
        y = empty_cells[0][choice]
        return self.new_play(y, x)
