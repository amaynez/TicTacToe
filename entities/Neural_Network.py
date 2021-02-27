import numpy as np
import json
from utilities import json_numpy
import utilities.constants as c


def activation(x, output=False):
    if output:
        if c.OUTPUT_ACTIVATION in ['sigmoid', 'Sigmoid', 'SIGMOID']:
            return 1 / (1 + np.exp(-x))
        elif c.OUTPUT_ACTIVATION in ['ReLU', 'relu', 'RELU']:
            return x * (x > 0)
    else:
        if c.ACTIVATION in ['sigmoid', 'Sigmoid', 'SIGMOID']:
            return 1 / (1 + np.exp(-x))
        elif c.ACTIVATION in ['ReLU', 'relu', 'RELU']:
            return x * (x > 0)


def d_activation(x, output=False):
    if output:
        if c.OUTPUT_ACTIVATION in ['sigmoid', 'Sigmoid', 'SIGMOID']:
            return x * (1 - x)
        elif c.OUTPUT_ACTIVATION in ['ReLU', 'relu', 'RELU']:
            return 1 * (x > 0)
    else:
        if c.ACTIVATION in ['sigmoid', 'Sigmoid', 'SIGMOID']:
            return x * (1 - x)
        elif c.ACTIVATION in ['ReLU', 'relu', 'RELU']:
            return 1 * (x > 0)

class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, learning_rate=0.1):
        self.inputs = inputs
        if isinstance(hidden, int):
            hidden = [hidden]
        self.hidden = np.array(hidden)
        self.outputs = outputs
        self.learning_rate = learning_rate

        # create weights and gradients for first hidden layer (defined based on number of inputs)
        self.weights = [np.random.uniform(-1, 1, size=(self.hidden[0], self.inputs)).astype(np.float128)]
        self.gradients = [np.zeros((self.hidden[0], self.inputs), np.float128)]

        # create weights and gradients for interior hidden layers (defined based on previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_col in enumerate(self.hidden[1:]):
                self.weights.append(np.random.uniform(-1, 1, size=(hidden_col, self.hidden[idx])).astype(np.float128))
                self.gradients.append(np.zeros((hidden_col, self.hidden[idx]), np.float128))

        # create weights and gradients for output layer (defined based on number of outputs)
        self.weights.append(np.random.uniform(-1, 1, size=(self.outputs, self.hidden[-1],)).astype(np.float128))
        self.gradients.append(np.zeros((self.outputs, self.hidden[-1],), np.float128))

        # create bias list of matrices (one per hidden layer and one for the output)
        self.bias = []
        self.bias_gradients = []
        for idx, hidden_col in enumerate(self.hidden):
            self.bias.append(np.random.uniform(-1, 1, size=(hidden_col, 1)).astype(np.float128))
            self.bias_gradients.append(np.zeros((hidden_col, 1), np.float128))
        self.bias.append(np.random.uniform(-1, 1, size=(self.outputs, 1)).astype(np.float128))
        self.bias_gradients.append(np.zeros((self.outputs, 1), np.float128))

        # create gradient matrices

    def gradient_zeros(self):
        self.gradients = [np.zeros((self.hidden[0], self.inputs), np.float128)]
        self.bias_gradients = [np.zeros((self.hidden[0], 1), np.float128)]
        if self.hidden.ndim > 0:
            for idx, hidden_col in enumerate(self.hidden[1:]):
                self.gradients.append(np.zeros((hidden_col, self.hidden[idx]), np.float128))
                self.bias_gradients.append(np.zeros((hidden_col, 1), np.float128))
        self.gradients.append(np.zeros((self.outputs, self.hidden[-1]), np.float128))
        self.bias_gradients.append(np.zeros((self.outputs, 1), np.float128))

    def copy_from(self, neural_net):
        self.weights = neural_net.weights
        self.bias = neural_net.bias

    def save_to_file(self, file_name='NeuralNet.json'):
        json_file = {
            'weights': self.weights,
            'biases': self.bias}
        with open(file_name, 'w') as file:
            json.dump(
                json_file,
                file,
                ensure_ascii=False,
                cls=json_numpy.EncodeFromNumpy)

    def load_from_file(self, file_name='NeuralNet.json'):
        with open(file_name) as file:
            json_file = json.load(file, cls=json_numpy.DecodeToNumpy)
        self.weights = json_file['weights']
        self.bias = json_file['biases']

    def forward_propagation(self, input_values, **kwargs):
        # create hidden results list for results matrices per hidden layer
        hidden_results = []

        # prepare the input values for matrix multiplication
        input_values = np.array(input_values)[np.newaxis].T

        # calculate results for the first hidden layer (depending on the inputs)
        hidden_results.append(activation(np.matmul(self.weights[0], input_values) + self.bias[0]))

        # calculate results for subsequent hidden layers if any (depending on the previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_cells in enumerate(self.hidden[1:]):
                hidden_results.append(activation(np.matmul(self.weights[idx + 1],
                                                           hidden_results[idx]) +
                                                 self.bias[idx + 1]))

        # calculate final result and return, if explicit is set then return all the intermediate results as well
        output = []
        if 'explicit' in kwargs.keys():
            if kwargs.get('explicit') in ['yes', 'y', 1]:
                output = hidden_results
        output.append(activation(np.matmul(self.weights[-1], hidden_results[-1]) + self.bias[-1], True))
        return output

    def train_once(self, inputs, targets):
        # get the results including the hidden layers' (intermediate results)
        results = self.forward_propagation(inputs, explicit='yes')

        # prepare the targets and inputs for matrix operations
        targets = np.array(targets)[np.newaxis].T
        input_values = np.array(inputs)[np.newaxis].T

        # calculate the error (outputs vs targets), index 0
        error = [results[-1] - targets]

        # calculate the error of the hidden layers from last to first but insert in the correct order
        for idx in range(len(results) - 2, -1, -1):
            error.insert(0, np.matmul(self.weights[idx + 1].T, error[0]))

        # modify weights and biases (input -> first hidden layer)
        self.weights[0] -= np.matmul((error[0] * d_activation(results[0]) * self.learning_rate), input_values.T)
        self.bias[0] -= (error[0] * d_activation(results[0])) * self.learning_rate

        # modify weights and biases (all subsequent hidden layers and output)
        for idx, weight_cols in enumerate(self.weights[1:-1]):
            weight_cols -= np.matmul((error[idx + 1] * d_activation(results[idx + 1]) * self.learning_rate),
                                     results[idx].T)
            self.bias[idx + 1] -= (error[idx + 1] * d_activation(results[idx + 1])) * self.learning_rate
        self.weights[-1] -= np.matmul((error[-1] * d_activation(results[-1], True) * self.learning_rate),
                                      results[-2].T)
        self.bias[-1] -= (error[-1] * d_activation(results[-1], True)) * self.learning_rate

    def calculate_gradient(self, inputs, targets):
        # get the results including the hidden layers' (intermediate results)
        results = self.forward_propagation(inputs, explicit='yes')

        # prepare the inputs for matrix operations
        input_values = np.array(inputs)[np.newaxis].T

        # calculate the error (outputs vs targets), index 0
        error = [results[-1] - targets]

        # calculate the error of the hidden layers from last to first but insert in the correct order
        for idx in range(len(results) - 2, -1, -1):
            error.insert(0, np.matmul(self.weights[idx + 1].T, error[0]))

        # modify weights and biases (input -> first hidden layer)
        self.gradients[0] -= np.matmul((error[0] * d_activation(results[0]) * self.learning_rate), input_values.T)
        self.bias_gradients[0] -= (error[0] * d_activation(results[0])) * self.learning_rate

        # modify weights and biases (all subsequent hidden layers and output)
        for idx, gradient_col in enumerate(self.gradients[1:-1]):
            gradient_col -= np.matmul((error[idx + 1] * d_activation(results[idx + 1]) * self.learning_rate),
                                      results[idx].T)
            self.bias_gradients[idx + 1] -= (error[idx + 1] * d_activation(results[idx + 1])) * self.learning_rate
        self.gradients[-1] -= np.matmul((error[-1] * d_activation(results[-1], True) * self.learning_rate),
                                        results[-2].T)
        self.bias_gradients[-1] -= (error[-1] * d_activation(results[-1], True)) * self.learning_rate

    def apply_gradients(self):
        for idx, gradient_col in enumerate(self.gradients):
            self.weights[idx] += gradient_col / c.BATCH_SIZE
            self.bias[idx] += self.bias_gradients[idx] / c.BATCH_SIZE
        self.gradient_zeros()

    def RL_train(self, replay_memory, target_network, experience):

        batch = experience(*zip(*replay_memory))
        states = np.array(batch.state)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward)
        next_states = np.array(batch.next_state)
        eps = np.finfo(np.float32).eps.item()
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        for i in range(len(replay_memory)):
            # calculate max_Q2
            if rewards[i] in (c.REWARD_LOST_GAME, c.REWARD_WON_GAME, c.REWARD_TIE_GAME):
                target_q = rewards[i]
            else:
                next_state = next_states[i].reshape(c.INPUTS)
                results = target_network.forward_propagation(next_state)
                results = results[0]
                max_q2 = np.max(results)
                target_q = rewards[i] + c.GAMMA * max_q2

            # form targets matrix
            state = states[i].reshape(c.INPUTS)
            targets = self.forward_propagation(state)
            targets = targets[0]
            targets[actions[i], 0] = target_q

            self.calculate_gradient(state, targets)

        print('.', end='')
        # for idx, gradient in enumerate(self.gradients):
        #     print('Gradients ', str(idx), ': ', round(gradient.sum(), 3))

        self.apply_gradients()
