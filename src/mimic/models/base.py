from abc import ABC, abstractmethod
import numpy as np
import networkx as nx


from ..fns import ActivationFunction, ErrorFunction, MSE, Sigmoid


class ANN(ABC):
    def __init__(
        self, layer_sizes, activation_fn: ActivationFunction, error_fn: ErrorFunction
    ):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.activations = []
        self.gradients = []

        self.activation_fn = activation_fn
        self.error_fn = error_fn

        # Initialize weights
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]))

        self._training_error = None

    @property
    def training_error(self):
        return self._training_error

    def set_training_error(self, value):
        self._training_error = value

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backward(self, output_data, learning_rate):
        pass

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, index):
        return self.weights[index]

    def to_networkx(self):
        G = nx.DiGraph()

        for i, layer_size in enumerate(self.layer_sizes):
            for j in range(layer_size):
                G.add_node((i, j), pos=(i, j))

        for i, weight in enumerate(self.weights):
            for j in range(weight.shape[0]):
                for k in range(weight.shape[1]):
                    G.add_edge((i, j), (i + 1, k), weight=weight[j][k])

        for i, activation in enumerate(self.activations):
            for j in range(activation.shape[0]):
                G.nodes[(i, j)]["activations"] = activation[j]

        return G
