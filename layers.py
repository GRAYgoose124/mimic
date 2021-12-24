import numpy as np

from numpy import vectorize
from utils import sigmoid, pd_sigmoid


class Layer:
    def __init__(self, width: int, squash_func: str, error_func: str):
        self.width = width
        self.nodes = np.zeros(width)
        self.weights = []
        self.squash = vectorize(sigmoid)
        self.error = vectorize(pd_sigmoid)
        self.connected_layers = {}

    def activate(self, input_data):
        raise NotImplementedError

    def connect(self, layers):
        raise NotImplementedError

       
class Dense(Layer):
    def __init__(self, width: int, squash_func='sigmoid', error_func='pd_sigmoid'):
        super().__init__(width, squash_func, error_func)

    def activate(self, input_data, update=False):
        activation = None
        # input layer
        if 'prev' not in self.connected_layers:
            activation = np.array(input_data)
        # hidden and output 
        else:
            print(self.connected_layers['prev'].weights)
            activation = self.squash(np.array(input_data.dot(self.connected_layers['prev'].weights)))

        if update:
            self.nodes = activation

        return activation

    def connect(self, next_layer, contype='full'):
        if contype == 'full':
            self.weights = np.ones(shape=(self.width, next_layer.width))
        elif contype == 'random':
            self.weights = np.random.random_sample((self.width, next_layer.width))

        self.connected_layers['next'] = next_layer
        next_layer.connected_layers['prev'] = self

    