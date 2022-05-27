import numpy as np

from numpy import vectorize
import utils


class Layer:
    def __init__(self, width: int, squash: callable = None, error: callable = None):
        self.width = width
        self.connected_layers = {}

        self.nodes = np.ones(width)
        self.weights = np.ones(width)
        self.errors = np.ones(width)

        if squash is None:
            squash = utils.sigmoid
        self.squash = vectorize(squash)

        if error is None:
            error = utils.pd_sigmoid
        self.error = (lambda x: vectorize(error)(x))(self.nodes)

    def activate(self, input_data):
        raise NotImplementedError

    def connect(self, layers):
        raise NotImplementedError

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
            activation = np.array(input_data)
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

        self.errors = np.zeros(shape=(next_layer.width, self.width))

        self.connected_layers['next'] = next_layer
        next_layer.connected_layers['prev'] = self

    