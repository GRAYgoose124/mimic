import numpy as np

from .layer import Layer


class Dense(Layer):
    def __init__(self, width: int, squash: callable = None, error: callable = None):
        super().__init__(width, squash, error)

    def activate(self, y_k, update):
        activation = None
        # input layer
        if 'prev' not in self.connected:
            activation = self.squash(np.array(y_k))
        # hidden and output 
        else:
            w_kj = self.connected['prev'].weights
            x_j = np.dot(y_k, w_kj)
            activation = self.squash(x_j)

        if update:
            self.nodes = activation

        return activation

    def reset(self):
        self.nodes = np.zeros(self.width)

    def connect(self, next_layer, contype='ones'):
        if contype == 'ones':
            self.weights = np.ones(shape=(self.width, next_layer.width))
        elif contype == 'random':
            self.weights = np.random.rand(self.width, next_layer.width)

        self.errors = np.zeros((next_layer.width, self.width))
        # self.errors = np.zeros((self.width, next_layer.width))

        self.connected['next'] = next_layer
        next_layer.connected['prev'] = self

    def error(self, expected=None):
        # hidden layers
        if 'prev' in self.connected and 'next' in self.connected:
            error_sum = np.dot(self.connected['next'].errors, self.weights.transpose())  # Scaling issue? checked with avg - no
            error_term = error_sum * self.nodes * (1 - self.nodes)

        # output layer
        elif 'next' not in self.connected and expected is not None:
            error_term = (expected - self.nodes) * (1 - self.nodes)

        self.errors = error_term
        #  return error_term
    