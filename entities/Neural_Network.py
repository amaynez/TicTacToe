import numpy as np
import json
from utilities import json_numpy
import utilities.constants as c


def activation(x, output=False):
    activation_function = c.OUTPUT_ACTIVATION if output else c.ACTIVATION
    if activation_function in ['sigmoid', 'Sigmoid', 'SIGMOID']:
        return 1 / (1 + np.exp(-x))
    elif activation_function in ['ReLU', 'relu', 'RELU']:
        return x * (x > 0)
    elif activation_function in ['linear', 'Linear', 'line']:
        return x


def d_activation(x, output=False):
    activation_function = c.OUTPUT_ACTIVATION if output else c.ACTIVATION
    if activation_function in ['sigmoid', 'Sigmoid', 'SIGMOID']:
        return x * (1 - x)
    elif activation_function in ['ReLU', 'relu', 'RELU']:
        return 1 * (x > 0)
    elif activation_function in ['linear', 'Linear', 'line']:
        return np.ones_like(x)


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

        # create bias and bias_gradients lists of matrices (one per hidden layer and one for the output)
        self.bias = []
        self.bias_gradients = []
        for idx, hidden_col in enumerate(self.hidden):
            self.bias.append(np.random.uniform(-1, 1, size=(hidden_col, 1)).astype(np.float128))
            self.bias_gradients.append(np.zeros((hidden_col, 1), np.float128))
        self.bias.append(np.random.uniform(-1, 1, size=(self.outputs, 1)).astype(np.float128))
        self.bias_gradients.append(np.zeros((self.outputs, 1), np.float128))

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
        try:
            with open(file_name, 'w') as file:
                json.dump(
                    json_file,
                    file,
                    ensure_ascii=False,
                    cls=json_numpy.EncodeFromNumpy)
                print('weights saved to file')
        except:
            print('cannot save to ', file)

    def load_from_file(self, file_name='NeuralNet.json'):
        try:
            with open(file_name) as file:
                json_file = json.load(file, cls=json_numpy.DecodeToNumpy)
                print('weights loaded from file')
                self.weights = json_file['weights']
                self.bias = json_file['biases']
        except:
            print('cannot open ', file_name)

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

        # calculate the derivative error_matrix (targets vs outputs), index 0
        d_error_matrix = [((targets - results[-1]) * d_activation(results[-1], True)) / c.BATCH_SIZE]

        # calculate the derivative error_matrix of the hidden layers from last to first but insert in the correct order
        for idx in range(len(results) - 1, 0, -1):
            d_error_matrix.insert(0, np.matmul(self.weights[idx].T, d_error_matrix[0] * d_activation(results[idx])))

        # calculate the gradient for the weights that feed the output layer
        self.weights[-1] -= self.learning_rate * np.matmul(d_error_matrix[-1], results[-2].T)
        self.bias[-1] -= self.learning_rate * d_error_matrix[-1]

        # calculate the gradient for all subsequent hidden layers
        for idx, weight_col in enumerate(self.weights[1:-1]):
            weight_col -= self.learning_rate * np.matmul(d_error_matrix[idx + 1], results[idx].T)
            self.bias[idx + 1] -= self.learning_rate * d_error_matrix[idx + 1]

        # calculate the gradient for the first layer weights (input -> first hidden layer)
        self.weights[0] -= self.learning_rate * np.matmul(d_error_matrix[0], input_values.T)
        self.bias[0] -= self.learning_rate * d_error_matrix[0]

    def calculate_gradient(self, inputs, targets):
        # get the results including the hidden layers' (intermediate results)
        results = self.forward_propagation(inputs, explicit='yes')

        # prepare the inputs for matrix operations
        input_values = np.array(inputs)[np.newaxis].T

        # calculate the derivative error_matrix (targets vs outputs), index 0
        d_error_matrix = [((results[-1] - targets) * d_activation(results[-1], True)) / c.BATCH_SIZE]

        # calculate the derivative error_matrix of the hidden layers from last to first but insert in the correct order
        for idx in range(len(results) - 1, 0, -1):
            d_error_matrix.insert(0, np.matmul(self.weights[idx].T, d_error_matrix[0] * d_activation(results[idx])))

        # calculate the gradient for all subsequent hidden layers
        for idx, gradient_col in enumerate(self.gradients[1:]):
            gradient_col += np.matmul(d_error_matrix[idx + 1], results[idx].T)
            self.bias_gradients[idx + 1] += d_error_matrix[idx + 1]

        # calculate the gradient for the first layer weights (input -> first hidden layer)
        self.gradients[0] += np.matmul(d_error_matrix[0], input_values.T)
        self.bias_gradients[0] += d_error_matrix[0]

    def cyclic_learning_rate(self, iteration):
        cycle = np.floor(1 + iteration / (2 * c.LR_STEP_SIZE))
        x = np.abs(iteration / c.LR_STEP_SIZE - 2 * cycle + 1)
        self.learning_rate = c.LEARNING_RATE + (c.MAX_LR - c.LEARNING_RATE) * np.maximum(0, (1 - x))

    def apply_gradients(self, iteration):
        self.cyclic_learning_rate(iteration)
        for idx, weight_col in enumerate(self.weights):
            weight_col -= self.learning_rate * np.array(self.gradients[idx])
            self.bias[idx] -= self.learning_rate * np.array(self.bias_gradients[idx])
        self.gradient_zeros()

    def RL_train(self, replay_memory, target_network, experience, iteration):
        loss = 0
        batch = experience(*zip(*replay_memory))
        states = np.array(batch.state)
        actions = np.array(batch.action)
        rewards = np.array(batch.reward)
        next_states = np.array(batch.next_state)
        eps = np.finfo(np.float32).eps.item()
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + eps)

        for i in range(len(replay_memory)):
            # calculate max_Q2
            if rewards[i] in (c.REWARD_LOST_GAME, c.REWARD_WON_GAME, c.REWARD_TIE_GAME):
                target_q = rewards_normalized[i]
            else:
                next_state = next_states[i].reshape(c.INPUTS)
                results_q2 = target_network.forward_propagation(next_state)
                results_q2 = results_q2[0]
                max_q2 = np.max(results_q2)
                target_q = rewards_normalized[i] + c.GAMMA * max_q2

            # form targets matrix
            state = states[i].reshape(c.INPUTS)
            targets = self.forward_propagation(state)
            targets = targets[0]
            old_targets = targets.copy()
            targets[actions[i], 0] = target_q
            for idx, position in enumerate(state):
                if position != 0:
                    targets[idx, 0] = 0
            loss += np.sum(((targets - old_targets)**2)/c.BATCH_SIZE)
            self.calculate_gradient(state, targets)

        print('.', end='')
        self.apply_gradients(iteration)
        return loss
