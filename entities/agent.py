import numpy as np
import random
import entities.Neural_Network as nn
import math
from utilities import constants as c


class Agent:
    def __init__(self, inputs, hidden_layers, outputs, **kwargs):
        self.PolicyNetwork = nn.NeuralNetwork(inputs, hidden_layers, outputs, c.LEARNING_RATE)
        if 'load' in kwargs.keys():
            if kwargs.get('load') in ['yes', 'y', 'YES', 'Y', 1]:
                self.PolicyNetwork.load_from_file()
            else:
                self.PolicyNetwork.load_from_file(kwargs.get('load'))
        self.TargetNetwork = nn.NeuralNetwork(inputs, hidden_layers, outputs, c.LEARNING_RATE)
        self.TargetNetwork.copy_from(self.PolicyNetwork)
        self.state = []
        self.action = 0
        self.reward = 0
        self.next_state = []
        self.adversary = 1
        self.NNet_player = 2
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

    def play(self, previous_turn, game, replay_memory, experience):
        termination_state = 0
        sprite_params = ()
        inputs = game.state
        inputs = inputs.reshape(c.INPUTS)
        results = self.PolicyNetwork.forward_propagation(inputs)
        results = results[0]
        state_before_action = game.state.copy()
        while previous_turn == game.turn:
            action, row, col = self.eGreedyStrategy(results, game)
            self.action = action
            termination_state, sprite_params = game.new_play(row, col)
            self.reward = self.calculate_reward(previous_turn, game.turn, game.winner)
            self.next_state = game.state
            replay_memory.push(
                experience(
                    state_before_action,
                    self.action,
                    self.reward,
                    self.next_state.copy()))
        return termination_state, sprite_params

    def eGreedyStrategy(self, results, game):
        exploration_rate = self.get_exploration_rate()
        self.current_step += 1
        # Exploration V.S Exploitation
        if np.random.rand() < exploration_rate:
            action = np.random.choice(c.OUTPUTS)
            row, col = self.split_rowcol(action)
        else:
            if np.max(results) > 0:
                action = np.argmax(results)
                row, col = self.split_rowcol(action)
                results[action] = 0
            else:
                empty_cells = np.where(game.state == 0)
                choice = random.choice(range(len(empty_cells[0])))
                col = empty_cells[1][choice]
                row = empty_cells[0][choice]
                action = self.combine_rowcol(row, col)
        return action, row, col

    def get_exploration_rate(self):
        return c.eEND + (c.eSTART - c.eEND) * \
               math.exp(-1. * self.current_step * c.eDECAY)

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
