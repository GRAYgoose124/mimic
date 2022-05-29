import numpy as np

from numpy import vectorize
import utils


class Layer:
    def __init__(self, width: int, squash: callable = None, errorf: callable = None):
        self.width = width
        self.connected_layers = {}

        self.nodes = np.ones(width)
        self.weights = np.ones(width)
        self.errors = np.ones(width)
        self.deltas = np.zeros(width)

        if squash is None:
            squash = utils.sigmoid
        self.squash = vectorize(squash)

        if errorf is None:
            errorf = utils.pd_sigmoid
        self.errorf = (lambda x: vectorize(errorf)(x))(self.nodes)

    def activate(self, input_data):
        raise NotImplementedError

    def connect(self, layers):
        raise NotImplementedError

    def error(self, expected=None):
        if 'prev' in self.connected_layers and 'next' in self.connected_layers:
            pd_sig = self.errorf()
            error_sum = np.multiply(self.connected_layers['next'].errors, layer.weights.transpose())  # Scaling issue? checked with avg - no
            error_sum = sum(error_sum)
            error_term = np.multiply(error_sum, layer_pd_sig)

        elif 'next' not in self.connected_layers:
            error_term = (expected - self.nodes) * self.errorf()

        self.errors = error_term
        return error_term

    def __repr__(self):
        return str(self.weights.round(2))

        # return " ".join([str(x) for x in zip(self.nodes.round(2), self.weights.round(2))])

    def __str__(self):
        return self.__repr__()
    

class Dense(Layer):
    def __init__(self, width: int, squash: callable = None, error: callable = None):
        super().__init__(width, squash, error)

    def activate(self, input_data, update):
        activation = None
        # input layer
        if 'prev' not in self.connected_layers:
            activation = self.squash(np.array(input_data))
        # hidden and output 
        else:
            activation = self.squash(np.dot(input_data, (self.connected_layers['prev'].weights)))

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

        self.connected_layers['next'] = next_layer
        next_layer.connected_layers['prev'] = self

    