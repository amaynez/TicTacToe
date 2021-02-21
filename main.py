import pygame as pyg
import numpy as np
import math
import sys


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
        global PLAYER
        game.state = np.zeros((3, 3), np.int8)
        for player in PLAYER:
            player.kill()
        PLAYER = []

    def new_play(self, turn, x, y):
        if self.state[x, y] == 0:
            self.state[x, y] = turn
            PLAYER.append(Player(x, y, turn))
            all_sprites.add(PLAYER[-1])
            winner = self.check_winner()
            if winner > 0:
                self.end_game(winner)
            return 2 if turn == 1 else 1
        return turn

    def check_winner(self):
        for i in range(3):
            for j in range(1, 3, 1):
                if np.all(self.state[i, :] == j) or np.all(self.state[:, i] == j):
                    return j
        for j in range(1, 3, 1):
            if np.all(self.state.diagonal() == j) or np.all(np.flipud(self.state).diagonal() == j):
                return j
        empty = np.where(self.state == 0)
        if len(empty[0]) == 0:
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
            draw_text(screen, "It was a tie! ", 22,
                      WIDTH / 2, HEIGHT / 2, YELLOW)
        else:
            draw_text(screen, "Player " + str(winner) + ' won', 22,
                      WIDTH / 2, HEIGHT / 2, YELLOW)
            Score[winner - 1] += 1
        draw_text(screen, "<Press any key to restart>", 18, WIDTH / 2, HEIGHT * 3 / 4, YELLOW)
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


class Player(pyg.sprite.Sprite):
    def __init__(self, x, y, turn):
        pyg.sprite.Sprite.__init__(self)
        self.image = pyg.transform.scale(images[turn-1], (round(WIDTH / 6), round(HEIGHT / 6)))
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH / 10 * (x * 2 + 3), HEIGHT / 10 * (y * 2 + 3))


WIDTH = 480
HEIGHT = 480
FPS = 30

# define colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREY = (185, 185, 185)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)

# initialize pygame and create window
pyg.init()
pyg.mixer.init()
screen = pyg.display.set_mode((WIDTH, HEIGHT))
pyg.display.set_caption("Tic Tac Toe")
clock = pyg.time.Clock()
all_sprites = pyg.sprite.Group()

# load images and font
images = [pyg.image.load('cross.png').convert(),
          pyg.image.load('circle.png').convert()]
font_name = pyg.font.match_font('Arial Bold')

# initialize game
game = Game()
PLAYER = []
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
    draw_text(screen, str(Score[0]), 64, WIDTH / 20 * 2.5, HEIGHT / 20 * 2.5, RED)
    draw_text(screen, str(Score[1]), 64, WIDTH / 20 * 17.5, HEIGHT / 20 * 2.5, CYAN)
    all_sprites.draw(screen)
    pyg.display.flip()

pyg.quit()
