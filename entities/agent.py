import numpy as np
import random
import entities.Neural_Network as nn
import math
from utilities import constants as c


class Agent:
    def __init__(self, inputs, hidden_layers, outputs, learning_rate, **kwargs):
        self.PolicyNetwork = nn.NeuralNetwork(inputs, hidden_layers, outputs, learning_rate)
        if 'load' in kwargs.keys():
            if kwargs.get('load') in ['yes', 'y', 'YES', 'Y', 1]:
                self.PolicyNetwork.load_from_file()
            else:
                self.PolicyNetwork.load_from_file(kwargs.get('load'))
        self.TargetNetwork = nn.NeuralNetwork(inputs, hidden_layers, outputs, learning_rate)
        self.TargetNetwork.copy_from(self.PolicyNetwork)
        self.adversary = 2 if c.NNET_PLAYER == 1 else 1
        self.NNet_player = c.NNET_PLAYER
        self.current_step = 0

    def save(self, file=''):
        if file == '':
            self.PolicyNetwork.save_to_file()
        else:
            self.PolicyNetwork.save_to_file(file)

    def load(self, file=''):
        if file == '':
            self.PolicyNetwork.load_from_file()
        else:
            self.PolicyNetwork.load_from_file(file)

    def play(self, previous_turn, game, replay_memory, experience, illegal_moves):
        termination_state = 0
        sprite_params = ()
        inputs = game.state
        inputs = c.one_hot(inputs)
        results = self.PolicyNetwork.forward_propagation(inputs)
        results = results[0]
        state_before_action = game.state.copy()
        while previous_turn == game.turn:
            action, row, col, random_move = self.eGreedyStrategy(results, game)
            termination_state, sprite_params = game.new_play(row, col)
            reward = self.calculate_reward(previous_turn, game.turn, game.winner)
            replay_memory.push(
                experience(
                    state_before_action,
                    action,
                    reward,
                    game.state.copy()))
            if previous_turn == game.turn and random_move == 0:
                illegal_moves += 1
        return termination_state, sprite_params, illegal_moves

    def play_visual(self, previous_turn, game):
        termination_state = 0
        sprite_params = ()
        inputs = game.state
        inputs = c.one_hot(inputs)
        results = self.PolicyNetwork.forward_propagation(inputs)
        results = results[0]
        while previous_turn == game.turn:
            action = np.argmax(results)
            row, col = self.split_rowcol(action)
            termination_state, sprite_params = game.new_play(row, col)
            results[action] = -math.inf
        return termination_state, sprite_params

    def eGreedyStrategy(self, results, game):
        random_move = 0
        exploration_rate = self.get_exploration_rate()
        self.current_step += 1
        # Exploration V.S Exploitation
        if np.random.rand() < exploration_rate:
            action = np.random.choice(c.OUTPUTS)
            row, col = self.split_rowcol(action)
            random_move = 1
        else:
            if np.max(results) > -math.inf:
                action = np.argmax(results)
                row, col = self.split_rowcol(action)
                results[action] = -math.inf
            else:
                empty_cells = np.where(game.state == 0)
                choice = random.choice(range(len(empty_cells[0])))
                col = empty_cells[1][choice]
                row = empty_cells[0][choice]
                action = self.combine_rowcol(row, col)
        return action, row, col, random_move

    def get_exploration_rate(self):
        return c.eEND + (c.eSTART - c.eEND) * \
               math.exp(-1 * self.current_step * c.eDECAY)

    @staticmethod
    def split_rowcol(action):
        row = math.floor(action / 3)
        col = action % 3
        return row, col

    @staticmethod
    def combine_rowcol(row, col):
        action = row * 3 + col
        return action

    def calculate_reward(self, previous_turn, turn, winner):
        if winner == 0:
            if previous_turn == turn:
                return c.REWARD_BAD_CHOICE
            return 0
        elif winner == self.adversary:
            return c.REWARD_LOST_GAME
        elif winner == self.NNet_player:
            return c.REWARD_WON_GAME
        elif winner == 3:
            return c.REWARD_TIE_GAME
        return 0
