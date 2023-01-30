from mimic.net_utils import pd_sigmoid, sigmoid
import numpy as np
import logging

from .layer import Layer


logger = logging.getLogger()


class Dense(Layer):
    def __init__(self, width: int, squash: callable = None, error: callable = None):
        super().__init__(width, squash, error)

    def activate(self, y_k, update=False):
        y_j = None

        # input layer
        if 'prev' not in self.connected:
            y_j = self.squash(np.array(y_k))
        # hidden and output 
        else:
            w_kj = self.connected['prev'].weights
            x_j = np.dot(y_k, w_kj)
            y_j = self.squash(x_j)

        if update:
            self.nodes = y_j

        return y_j

    def connect(self, next_layer: Layer, contype: str = 'ones'):
        """
        Connect this layer to the next layer, densely and using the specified connection type.
        
        Parameters
        ----------
        next_layer : Layer
            The next layer to connect to
        contype : str, optional
            The type of connection to make, by default 'ones'
            - 'ones' : fully connect with ones
            - 'zeros' : fully connect with zeros
            - 'random' : fully connect with random weights
        """
        if contype == 'ones':
            self.weights = np.ones(shape=(self.width, next_layer.width))
        elif contype == 'random':
            self.weights = np.random.rand(self.width, next_layer.width)

        # self.errors = np.zeros((next_layer.width, self.width))
        self.errors = np.zeros((self.width, next_layer.width))

        self.connected['next'] = next_layer
        next_layer.connected['prev'] = self

    def error(self, expected=None, update=True):
        # hidden layers
        if 'prev' in self.connected and 'next' in self.connected:
            # error_sum = np.dot(self.connected['next'].errors, self.weights.transpose())  # Scaling issue? checked with avg - no
            error_sum = np.dot(self.weights, self.connected['next'].errors)
            error_term = error_sum * self.errorf(self.nodes)

        # output layer # this is assuming pd_sigmoid network
        elif 'next' not in self.connected and expected is not None:
            error_term = (expected - self.nodes) * (1 - self.nodes)

        if update:
            self.errors = error_term
        
        return error_term
    