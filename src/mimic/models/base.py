from abc import ABC, abstractmethod
import numpy as np
import networkx as nx


from ..utils.fns import ActivationFunction, ErrorFunction, MSE, Sigmoid


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

        self._G = None

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
        if self._G is None:
            # create if not yet created
            G = nx.DiGraph()

            # add nodes
            for i, layer_size in enumerate(self.layer_sizes):
                for j in range(layer_size):
                    G.add_node((i, j), pos=(i, j))

            # add edges with initialized weights
            for i, weight in enumerate(self.weights):
                for j, k in np.ndindex(weight.shape):
                    G.add_edge((i, j), (i + 1, k), weight=weight[j][k])

            self._G = G
        else:
            # just update weights
            for i, weight in enumerate(self.weights):
                for j, k in np.ndindex(weight.shape):
                    self._G.edges[(i, j), (i + 1, k)]["weight"] = weight[j][k]

        # update activations
        for i, activation in enumerate(self.activations):
            for j in range(activation.shape[0]):
                self._G.nodes[(i, j)]["activations"] = activation[j]

        return self._G
