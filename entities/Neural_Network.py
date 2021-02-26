import numpy as np
import json
from utilities import json_numpy
import utilities.constants as c


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, inputs, hidden, outputs, learning_rate=0.1):
        self.inputs = inputs
        if isinstance(hidden, int):
            hidden = [hidden]
        self.hidden = np.array(hidden)
        self.outputs = outputs
        self.learning_rate = learning_rate

        # create weights and gradients for first hidden layer (defined based on number of inputs)
        self.weights = [np.random.uniform(-1, 1, size=(self.hidden[0], self.inputs))]
        self.gradients = [np.zeros((self.hidden[0], self.inputs))]

        # create weights and gradients for interior hidden layers (defined based on previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_col in enumerate(self.hidden[1:]):
                self.weights.append(np.random.uniform(-1, 1, size=(hidden_col, self.hidden[idx])))
                self.gradients.append(np.zeros((hidden_col, self.hidden[idx])))

        # create weights and gradients for output layer (defined based on number of outputs)
        self.weights.append(np.random.uniform(-1, 1, size=(self.outputs, self.hidden[-1],)))
        self.gradients.append(np.zeros((self.outputs, self.hidden[-1],)))

        # create bias list of matrices (one per hidden layer and one for the output)
        self.bias = []
        self.bias_gradients = []
        for idx, hidden_col in enumerate(self.hidden):
            self.bias.append(np.random.uniform(-1, 1, size=(hidden_col, 1)))
            self.bias_gradients.append(np.zeros((hidden_col, 1)))
        self.bias.append(np.random.uniform(-1, 1, size=(self.outputs, 1)))
        self.bias_gradients.append(np.zeros((self.outputs, 1)))

        # create gradient matrices

    def gradient_zeros(self):
        self.gradients = [np.zeros((self.hidden[0], self.inputs))]
        self.bias_gradients = [np.zeros((self.hidden[0], 1))]
        if self.hidden.ndim > 0:
            for idx, hidden_col in enumerate(self.hidden[1:]):
                self.gradients.append(np.zeros((hidden_col, self.hidden[idx])))
                self.bias_gradients.append(np.zeros((hidden_col, 1)))
        self.gradients.append(np.zeros((self.outputs, self.hidden[-1])))
        self.bias_gradients.append(np.zeros((self.outputs, 1)))

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
        hidden_results.append(sigmoid(np.matmul(self.weights[0], input_values) + self.bias[0]))

        # calculate results for subsequent hidden layers if any (depending on the previous layer)
        if self.hidden.ndim > 0:
            for idx, hidden_cells in enumerate(self.hidden[1:]):
                hidden_results.append(sigmoid(np.matmul(self.weights[idx + 1],
                                                        hidden_results[idx]) +
                                              self.bias[idx + 1]))

        # calculate final result and return, if explicit is set then return all the intermediate results as well
        output = []
        if 'explicit' in kwargs.keys():
            if kwargs.get('explicit') in ['yes', 'y', 1]:
                output = hidden_results
        output.append(sigmoid(np.matmul(self.weights[-1], hidden_results[-1]) + self.bias[-1]))
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
        self.weights[0] -= np.matmul((error[0] * d_sigmoid(results[0]) * self.learning_rate), input_values.T)
        self.bias[0] -= (error[0] * d_sigmoid(results[0])) * self.learning_rate

        # modify weights and biases (all subsequent hidden layers and output)
        for idx, weight_cols in enumerate(self.weights[1:]):
            weight_cols -= np.matmul((error[idx + 1] * d_sigmoid(results[idx + 1]) * self.learning_rate),
                                     results[idx].T)
            self.bias[idx + 1] -= (error[idx + 1] * d_sigmoid(results[idx + 1])) * self.learning_rate

    def calculate_gradient(self, inputs, targets):
        # get the results including the hidden layers' (intermediate results)
        results = self.forward_propagation(inputs, explicit='yes')

        # prepare the targets and inputs for matrix operations
        # targets = np.array(targets)[np.newaxis].T
        input_values = np.array(inputs)[np.newaxis].T

        # calculate the error (outputs vs targets), index 0
        error = [results[-1] - targets]

        # calculate the error of the hidden layers from last to first but insert in the correct order
        for idx in range(len(results) - 2, -1, -1):
            error.insert(0, np.matmul(self.weights[idx + 1].T, error[0]))

        # modify weights and biases (input -> first hidden layer)
        self.gradients[0] -= np.matmul((error[0] * d_sigmoid(results[0]) * self.learning_rate), input_values.T)
        self.bias_gradients[0] -= (error[0] * d_sigmoid(results[0])) * self.learning_rate

        # modify weights and biases (all subsequent hidden layers and output)
        for idx, gradient_col in enumerate(self.gradients[1:]):
            gradient_col -= np.matmul((error[idx + 1] * d_sigmoid(results[idx + 1]) * self.learning_rate),
                                     results[idx].T)
            self.bias_gradients[idx + 1] -= (error[idx + 1] * d_sigmoid(results[idx + 1])) * self.learning_rate

    def apply_gradients(self):
        for idx, gradient_col in enumerate(self.gradients):
            self.weights[idx] += gradient_col / c.BATCH_SIZE
            self.bias[idx] += self.bias_gradients[idx] / c.BATCH_SIZE
        self.gradient_zeros()

    def RL_train(self, replay_memory, target_network):
        for experience in replay_memory:
            # calculate max_Q2
            experience.next_state = experience.next_state.reshape(self.inputs)
            experience.state = experience.state.reshape(self.inputs)
            results = target_network.forward_propagation(experience.next_state)
            results = results[0]
            max_q2 = np.max(results)

            target_q = experience.reward + c.GAMMA*max_q2

            # form targets matrix
            targets = self.forward_propagation(experience.state)
            targets = targets[0]
            targets[experience.action, 0] = target_q

            self.calculate_gradient(experience.state, targets)
        print('.', end='')
        self.apply_gradients()
