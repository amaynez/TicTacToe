import numpy as np
import random
import utilities.constants as c


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

    @staticmethod
    def return_winner(state):
        winner = 0
        for row_col in range(3):
            for player in range(1, 3, 1):
                if np.all(state[row_col, :] == player) or np.all(state[:, row_col] == player):
                    winner = player
        for player in range(1, 3, 1):
            if np.all(state.diagonal() == player) or np.all(np.flipud(state).diagonal() == player):
                winner = player
        if winner == 0:
            empty_cells = np.where(state == 0)
            if len(empty_cells[0]) == 0:
                winner = 3
        return winner

    def minmax_play(self):
        row_to_play, col_to_play, depth = 0, 0, 0
        max_value = - np.inf
        empty_cells = self.get_empty_cells(self.state)
        for i, row in enumerate(empty_cells[0]):
            state = self.state.copy()
            state[row, empty_cells[1][i]] = self.turn
            value = self.best_move(state, self.switch_turn(self.turn), depth)
            if value > max_value:
                max_value = value
                row_to_play = row
                col_to_play = empty_cells[1][i]
                if max_value == 10:
                    return row_to_play, col_to_play
        return row_to_play, col_to_play

    def best_move(self, state, turn, depth):
        winner = self.return_winner(state)
        multiplier = -1 if depth % 2 == 0 else 1
        value = self.assign_value(winner)
        if winner > 0:
            return value
        compare = - np.inf * multiplier
        empty_cells = self.get_empty_cells(state)
        for i, row in enumerate(empty_cells[0]):
            next_state = state.copy()
            next_state[row, empty_cells[1][i]] = turn
            value = self.best_move(next_state, self.switch_turn(turn), depth + 1)
            if multiplier == 1:
                compare = np.maximum(value, compare)
                if compare == 10:
                    return compare
            elif multiplier == -1:
                compare = np.minimum(value, compare)
                if compare == -10:
                    return compare
        return compare

    @staticmethod
    def get_empty_cells(state):
        return np.where(state == 0)

    @staticmethod
    def assign_value(winner):
        if winner == 3 or winner == 0:
            return 0
        elif winner == 2:
            return 10
        return -10

    @staticmethod
    def switch_turn(turn):
        return 2 if turn == 1 else 1

    def AI_play(self):
        if c.AI_ENGINE == 'minimax':
            row, col = self.minmax_play()
        elif c.AI_ENGINE == 'hardcode':
            row, col = self.hard_coded_AI()
        else:
            row, col = self.random_play()
        return self.new_play(row, col)

    def hard_coded_AI(self):
        adversary = 1 if self.turn == 2 else 2
        # finish immediate threat from us (win)
        # and block immediate threats from adversary (don't loose)
        for player in range(self.turn, 3 if self.turn == 1 else 0, 1 if self.turn == 1 else -1):
            for row_col in range(3):
                for col_row in range(3):
                    # check rows
                    if self.state[row_col, col_row] == 0 and \
                            np.all(self.state[row_col, (np.arange(3) != col_row)] == player):
                        return row_col, col_row
                    # check cols
                    if self.state[col_row, row_col] == 0 and \
                            np.all(self.state[(np.arange(3) != col_row), row_col] == player):
                        return col_row, row_col
                # check main diagonal
                diagonal = np.diagonal(self.state)
                if self.state[row_col, row_col] == 0 and \
                        np.all(diagonal[(np.arange(3) != row_col)] == player):
                    return row_col, row_col
                # check secondary diagonal
                diagonal = np.diagonal(np.flipud(self.state))
                if self.state[2 - row_col, row_col] == 0 and \
                        np.all(diagonal[(np.arange(3) != row_col)] == player):
                    return 2 - row_col, row_col

        # else check for corner threats
        for i in range(2):
            if self.state[0, i * 2] == adversary and \
                    self.state[2, 2 - (i * 2)] == adversary and \
                    np.all(self.state[0, (1 - i, 2 - i)] == 0) and \
                    np.all(self.state[(i, 1 + i), 2] == 0) and \
                    np.all(self.state[(1 - i, 2 - i), 0] == 0) and \
                    np.all(self.state[2, (i, 1 + i)]) == 0:
                return 0, 1

        # else check for center corner threat
        if self.state[0, 2] == 0 and \
                self.state[1, 1] == adversary and \
                self.state[2, 2] == adversary:
            return 0, 2

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
                                    return row_col, i

        # else play center
        if self.state[1, 1] == 0:
            return 1, 1

        # else play corner
        for y in range(0, 3, 2):
            for x in range(0, 3, 2):
                if self.state[y, x] == 0:
                    return y, x

        # else play random move from the available cells
        return self.random_play()

    def random_play(self):
        empty_cells = np.where(self.state == 0)
        choice = random.choice(range(len(empty_cells[0])))
        x = empty_cells[1][choice]
        y = empty_cells[0][choice]
        return y, x
