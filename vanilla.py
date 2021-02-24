import pygame as pyg
import numpy as np
import math
import sys
import random


WIDTH = 480
HEIGHT = 480
FPS = 30

BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREY = (185, 185, 185)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)


def draw_text(surf, text, size, x, y, color=(255, 255, 255)):
    font = pyg.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    surf.blit(text_surface, text_rect)


def draw_background():
    for i in range(2, 4, 1):
        pyg.draw.line(screen, GREY,
                      (WIDTH / 5 * i, HEIGHT / 5),
                      (WIDTH / 5 * i, HEIGHT / 5 * 4), 11)
        pyg.draw.line(screen, GREY,
                      (WIDTH / 5, HEIGHT / 5 * i),
                      (WIDTH / 5 * 4, HEIGHT / 5 * i), 11)


class Game:

    def __init__(self):
        self.state = np.zeros((3, 3), np.int8)

    def new_game(self):
        global PLAY_SPRITES
        game.state = np.zeros((3, 3), np.int8)
        for player in PLAY_SPRITES:
            player.kill()
        PLAYER = []

    def new_play(self, turn, x, y):
        if self.state[y, x] == 0:
            self.state[y, x] = turn
            PLAY_SPRITES.append(PlayerSprite(x, y, turn))
            all_sprites.add(PLAY_SPRITES[-1])
            winner = self.check_winner()
            if winner > 0:
                self.end_game(winner)
            return 2 if turn == 1 else 1
        return turn

    def check_winner(self):
        for row_col in range(3):
            for player in range(1, 3, 1):
                if np.all(self.state[row_col, :] == player) or np.all(self.state[:, row_col] == player):
                    return player
        for player in range(1, 3, 1):
            if np.all(self.state.diagonal() == player) or np.all(np.flipud(self.state).diagonal() == player):
                return player
        empty_cells = np.where(self.state == 0)
        if len(empty_cells[0]) == 0:
            self.end_game(0)
        return 0

    def end_game(self, winner):
        global Score
        all_sprites.draw(screen)
        s = pyg.Surface((WIDTH, HEIGHT), pyg.SRCALPHA)
        s.fill((64, 64, 64, 164))
        screen.blit(s, (0, 0))
        draw_text(screen, "Game Over!", 64, WIDTH / 2, HEIGHT / 4, YELLOW)
        if winner == 0:
            draw_text(screen, "It was a tie! ", 32,
                      WIDTH / 2, HEIGHT / 2, YELLOW)
        else:
            draw_text(screen, "Player " + str(winner) + ' won', 32,
                      WIDTH / 2, HEIGHT / 2, RED if winner == 1 else CYAN)
            Score[winner - 1] += 1
        draw_text(screen, "<Press any key to restart>", 24, WIDTH / 2, HEIGHT * 3 / 4, YELLOW)
        pyg.display.flip()
        waiting = True
        while waiting:
            clock.tick(FPS)
            for event in pyg.event.get():
                if event.type == pyg.QUIT:
                    pyg.quit()
                    sys.exit()
                if event.type == pyg.KEYDOWN or event.type == pyg.MOUSEBUTTONDOWN:
                    key_state = pyg.key.get_pressed()
                    if key_state[pyg.K_ESCAPE]:
                        pyg.quit()
                        sys.exit()
                    waiting = False
                    self.new_game()

    def AI_play(self, turn):
        adversary = 1 if turn == 2 else 2

        # finish immediate threat from us (win)
        # and block immediate threats from adversary (don't loose)
        for player in range(2, 0, -1):
            for row_col in range(3):
                for col_row in range(3):
                    # check rows
                    if self.state[row_col, col_row] == 0 and \
                            np.all(self.state[row_col, (np.arange(3) != col_row)] == player):
                        turn = self.new_play(turn, col_row, row_col)
                        return turn
                    # check cols
                    if self.state[col_row, row_col] == 0 and \
                            np.all(self.state[(np.arange(3) != col_row), row_col] == player):
                        turn = self.new_play(turn, row_col, col_row)
                        return turn
                # check main diagonal
                diagonal = np.diagonal(self.state)
                if self.state[row_col, row_col] == 0 and \
                        np.all(diagonal[(np.arange(3) != row_col)] == player):
                    turn = self.new_play(turn, row_col, row_col)
                    return turn
                # check secondary diagonal
                diagonal = np.diagonal(np.flipud(self.state))
                if self.state[2-row_col, row_col] == 0 and \
                        np.all(diagonal[(np.arange(3) != row_col)] == player):
                    turn = self.new_play(turn, row_col, 2-row_col)
                    return turn

        # else check for corner threats
        for i in range(2):
            if self.state[0, i*2] == adversary and \
                    self.state[2, 2-(i*2)] == adversary and \
                    np.all(self.state[0, (1-i, 2-i)] == 0) and \
                    np.all(self.state[(i, 1+i), 2] == 0) and \
                    np.all(self.state[(1-i, 2-i), 0] == 0) and \
                    np.all(self.state[2, (i, 1+i)]) == 0:
                turn = self.new_play(turn, 1, 0)
                return turn

        # else check for center corner threat
        if self.state[0, 2] == 0 and \
            self.state[1, 1] == adversary and \
                self.state[2, 2] == adversary:
            turn = self.new_play(turn, 2, 0)
            return turn

        # else go for possible simultaneous threat from us
        # and block possible simultaneous threats from adversary
        for player in range(2, 0, -1):
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
                                    turn = self.new_play(turn, i, row_col)
                                    return turn

        # else play center
        if self.state[1, 1] == 0:
            turn = self.new_play(turn, 1, 1)
            return turn

        # else play corner
        for y in range(0, 3, 2):
            for x in range(0, 3, 2):
                if self.state[y, x] == 0:
                    turn = self.new_play(turn, x, y)
                    return turn

        # else play random move from the available cells
        empty_cells = np.where(self.state == 0)
        choice = random.choice(range(len(empty_cells[0])))
        x = empty_cells[1][choice]
        y = empty_cells[0][choice]
        turn = self.new_play(turn, x, y)
        return turn


class PlayerSprite(pyg.sprite.Sprite):
    def __init__(self, x, y, turn):
        pyg.sprite.Sprite.__init__(self)
        self.image = pyg.transform.scale(images[turn - 1], (round(WIDTH / 6), round(HEIGHT / 6)))
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH / 10 * (x * 2 + 3), HEIGHT / 10 * (y * 2 + 3))


# initialize pygame and create window
pyg.init()
pyg.mixer.init()
screen = pyg.display.set_mode((WIDTH, HEIGHT))
pyg.display.set_caption("Tic Tac Toe")
clock = pyg.time.Clock()
all_sprites = pyg.sprite.Group()

# load images and font
images = [pyg.image.load('media/cross.png').convert(),
          pyg.image.load('media/circle.png').convert()]
font_name = pyg.font.match_font('Calibri')

# initialize game
game = Game()
PLAY_SPRITES = []
turn = 1
Score = [0, 0]

# Game loop
running = True
while running:
    clock.tick(FPS)
    for event in pyg.event.get():
        if event.type == pyg.QUIT:
            running = False
        elif event.type == pyg.MOUSEBUTTONDOWN:
            x, y = pyg.mouse.get_pos()
            if (WIDTH / 5) < x < (WIDTH / 5 * 4) and (HEIGHT / 5) < y < (HEIGHT / 5 * 4):
                turn = game.new_play(turn,
                                     math.floor(x / (WIDTH / 5) - 1),
                                     math.floor(y / (HEIGHT / 5)) - 1)
        elif event.type == pyg.KEYDOWN:
            key_state = pyg.key.get_pressed()
            if key_state[pyg.K_ESCAPE]:
                running = False

    # Update
    all_sprites.update()

    # Draw / render
    screen.fill(BLACK)
    draw_background()
    if turn == 1:
        pyg.draw.rect(screen, RED,
                      (round(WIDTH / 20),
                       round(HEIGHT / 20),
                       round(WIDTH / 20 * 3),
                       round(HEIGHT / 20 * 3)),
                      5,
                      8)
    else:
        pyg.draw.rect(screen, CYAN,
                      (round(WIDTH / 20 * 16),
                       round(HEIGHT / 20),
                       round(WIDTH / 20 * 3),
                       round(HEIGHT / 20 * 3)),
                      5,
                      8)
        turn = game.AI_play(turn)
    draw_text(screen, str(Score[0]), 64, WIDTH / 20 * 2.5, HEIGHT / 20 * 2.5, RED)
    draw_text(screen, str(Score[1]), 64, WIDTH / 20 * 17.5, HEIGHT / 20 * 2.5, CYAN)
    all_sprites.draw(screen)
    pyg.display.flip()

pyg.quit()
