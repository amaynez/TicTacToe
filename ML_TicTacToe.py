import pygame as pyg
import math
import sys
import Game
import agent
import constants as c
import Memory
from recordtype import recordtype

experience = recordtype('experience', 'state action reward next_state')


class PlayerSprite(pyg.sprite.Sprite):
    def __init__(self, row, col, turn):
        pyg.sprite.Sprite.__init__(self)
        self.image = pyg.transform.scale(images[turn - 1], (round(c.WIDTH / 6), round(c.HEIGHT / 6)))
        self.image.set_colorkey(c.BLACK)
        self.rect = self.image.get_rect()
        self.rect.center = (c.WIDTH / 10 * (col * 2 + 3), c.HEIGHT / 10 * (row * 2 + 3))


def draw_text(surf, text, size, x, y, color=(255, 255, 255)):
    font = pyg.font.Font(font_name, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    surf.blit(text_surface, text_rect)


def draw_background():
    for i in range(2, 4, 1):
        pyg.draw.line(screen, c.GREY,
                      (c.WIDTH / 5 * i, c.HEIGHT / 5),
                      (c.WIDTH / 5 * i, c.HEIGHT / 5 * 4), 11)
        pyg.draw.line(screen, c.GREY,
                      (c.WIDTH / 5, c.HEIGHT / 5 * i),
                      (c.WIDTH / 5 * 4, c.HEIGHT / 5 * i), 11)


def game_over_screen(winner):
    s = pyg.Surface((c.WIDTH, c.HEIGHT), pyg.SRCALPHA)
    s.fill((64, 64, 64, 164))
    screen.blit(s, (0, 0))
    draw_text(screen, "Game Over!", 64, c.WIDTH / 2, c.HEIGHT / 4, c.YELLOW)
    if winner == 3:
        draw_text(screen, "It was a tie! ", 32,
                  c.WIDTH / 2, c.HEIGHT / 2, c.YELLOW)
    else:
        draw_text(screen, "Player " + str(winner) + ' won', 32,
                  c.WIDTH / 2, c.HEIGHT / 2, c.RED if winner == 1 else c.CYAN)
        game.score[winner - 1] += 1
    draw_text(screen, "<Press any key to restart>", 24, c.WIDTH / 2, c.HEIGHT * 3 / 4, c.YELLOW)
    pyg.display.flip()
    waiting = True
    while waiting:
        clock.tick(c.FPS)
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
                game.new_game()


# initialize pygame and create window
pyg.init()
pyg.mixer.init()
screen = pyg.display.set_mode((c.WIDTH, c.HEIGHT))
pyg.display.set_caption("Tic Tac Toe")
clock = pyg.time.Clock()
all_sprites = pyg.sprite.Group()

# load images and font
images = [pyg.image.load('cross.png').convert(),
          pyg.image.load('circle.png').convert()]
font_name = pyg.font.match_font('Calibri')

# initialize game
game = Game.Game()

# create neural network
agent = agent.Agent(c.INPUTS, c.HIDDEN_LAYERS, c.OUTPUTS)
replay_memory = Memory.ReplayMemory(c.MEMORY_CAPACITY)

# Game loop
running = True
while running:
    sprite_params = ()
    termination_state = 0
    clock.tick(c.FPS)
    for event in pyg.event.get():
        if event.type == pyg.QUIT:
            running = False
        elif event.type == pyg.MOUSEBUTTONDOWN:
            x, y = pyg.mouse.get_pos()
            if (c.WIDTH / 5) < x < (c.WIDTH / 5 * 4) and (c.HEIGHT / 5) < y < (c.HEIGHT / 5 * 4):
                col = math.floor(x / (c.WIDTH / 5) - 1)
                row = math.floor(y / (c.HEIGHT / 5) - 1)
                termination_state, sprite_params = game.new_play(col, row)
                game.PLAY_SPRITES.append(PlayerSprite(sprite_params[0], sprite_params[1], sprite_params[2]))
                all_sprites.add(game.PLAY_SPRITES[-1])
        elif event.type == pyg.KEYDOWN:
            key_state = pyg.key.get_pressed()
            if key_state[pyg.K_ESCAPE]:
                running = False
    # Update
    all_sprites.update()

    # Draw / render
    screen.fill(c.BLACK)
    draw_background()
    previous_turn = game.turn
    if game.turn == 1:
        pyg.draw.rect(screen, c.RED,
                      (round(c.WIDTH / 20),
                       round(c.HEIGHT / 20),
                       round(c.WIDTH / 20 * 3),
                       round(c.HEIGHT / 20 * 3)),
                      5,
                      8)
        state_before_action = game.state.copy()
        while previous_turn == game.turn:
            termination_state, sprite_params = agent.play(previous_turn, game)
            replay_memory.memory.append(experience(state_before_action, agent.action, agent.reward, agent.next_state.copy()))
    else:
        pyg.draw.rect(screen, c.CYAN,
                      (round(c.WIDTH / 20 * 16),
                       round(c.HEIGHT / 20),
                       round(c.WIDTH / 20 * 3),
                       round(c.HEIGHT / 20 * 3)),
                      5,
                      8)
        termination_state, sprite_params = game.AI_play()
        if game.winner > 0:
            replay_memory.memory[-1].reward = agent.calculate_reward(previous_turn, game.turn, game.winner)
        if len(replay_memory.memory) > 0:
            replay_memory.memory[-1].next_state = game.state.copy()
    game.PLAY_SPRITES.append(PlayerSprite(sprite_params[0], sprite_params[1], sprite_params[2]))
    all_sprites.add(game.PLAY_SPRITES[-1])

    draw_text(screen, str(game.score[0]), 64, c.WIDTH / 20 * 2.5, c.HEIGHT / 20 * 2.5, c.RED)
    draw_text(screen, str(game.score[1]), 64, c.WIDTH / 20 * 17.5, c.HEIGHT / 20 * 2.5, c.CYAN)
    all_sprites.draw(screen)
    pyg.display.flip()

    if termination_state == -1:
        game_over_screen(game.winner)
        agent.train_NNet(replay_memory)
        replay_memory = Memory.ReplayMemory(c.MEMORY_CAPACITY)

pyg.quit()
