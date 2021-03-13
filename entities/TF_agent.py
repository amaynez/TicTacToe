import numpy as np
import random
import math
from utilities import constants as c
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model


class Agent:
    def __init__(self, inputs, hidden_layers, outputs):
        self.PolicyNetwork = Sequential()
        for layer in hidden_layers:
            self.PolicyNetwork.add(Dense(units=layer,
                                         activation='relu',
                                         input_dim=inputs,
                                         kernel_initializer='random_uniform',
                                         bias_initializer='zeros'))
        self.PolicyNetwork.add(Dense(outputs,
                                     kernel_initializer='random_uniform',
                                     bias_initializer='zeros'))
        self.PolicyNetwork.compile(optimizer='adam',
                                   loss='mean_squared_error',
                                   metrics=['accuracy'])
        try:
            self.PolicyNetwork = load_model('NNetTicTacToe.h5')
            print('Pre-existing model found... loading data.')
        except:
            pass

        self.TargetNetwork = Sequential()
        for layer in hidden_layers:
            self.TargetNetwork.add(Dense(units=layer,
                                         activation='relu',
                                         input_dim=inputs,
                                         kernel_initializer='random_uniform',
                                         bias_initializer='zeros'))
        self.TargetNetwork.add(Dense(outputs,
                                     kernel_initializer='random_uniform',
                                     bias_initializer='zeros'))
        self.TargetNetwork.compile(optimizer='adam',
                                   loss='mean_squared_error',
                                   metrics=['accuracy'])
        self.copy_target_network()
        self.adversary = 2 if c.NNET_PLAYER == 1 else 1
        self.NNet_player = c.NNET_PLAYER
        self.current_step = 0

    def copy_target_network(self):
        self.PolicyNetwork.save('_temp.h5')
        self.TargetNetwork = load_model('_temp.h5')

    def save_to_file(self):
        self.PolicyNetwork.save('NNetTicTacToe.h5')

    def play(self, previous_turn, game, replay_memory, experience, illegal_moves):
        termination_state = 0
        sprite_params = ()
        inputs = game.state
        inputs = c.one_hot(inputs)
        results = self.PolicyNetwork.predict(np.asarray([inputs]), batch_size=1)[0]
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
        results = self.PolicyNetwork.predict(np.asarray([inputs]), batch_size=1)[0]
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

    def RL_train(self, replay_memory, experience):
        loss = 0
        states_to_train = []
        targets_to_train = []
        batch = experience(*zip(*replay_memory))
        states = np.array(batch.state)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward)
        next_states = np.array(batch.next_state)
        eps = np.finfo(np.float32).eps.item()

        if c.REWARD_NORMALIZATION:
            rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + eps)
        else:
            rewards_normalized = rewards

        for i in range(len(replay_memory)):
            empty_cells = np.where(next_states[i] == 0)
            if len(empty_cells[0]) == 0:
                target_q = rewards_normalized[i]
            else:
                # calculate max_Q2
                next_state = c.one_hot(next_states[i])
                results_q2 = self.TargetNetwork.predict(np.asarray([next_state]), batch_size=1)[0]
                max_q2 = np.max(results_q2)
                target_q = rewards_normalized[i] + c.GAMMA * max_q2

            # form targets vector
            state = c.one_hot(states[i])
            targets = self.PolicyNetwork.predict(np.asarray([state]), batch_size=1)[0]
            targets[actions[i]] = target_q
            states_to_train.append(state)
            targets_to_train.append(targets)

        history = self.PolicyNetwork.fit(np.asarray(states_to_train),
                                         np.asarray(targets_to_train),
                                         epochs=1,
                                         batch_size=c.BATCH_SIZE,
                                         verbose=2)
        self.save_to_file()
        return history
