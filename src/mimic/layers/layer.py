import numpy as np

from numpy import vectorize

from ..utils.net import sigmoid, pd_sigmoid


class Layer:
    def __init__(self, width: int, squash: callable = None, errorf: callable = None):
        self.width = width
        self.connected = {}

        self.nodes = np.zeros(width)
        self.weights = np.ones(width)
        self.errors = np.zeros(width)
        self.deltas = np.zeros(width)

        self.shape = (width,)

        if squash is None:
            squash = sigmoid
        self.squash = vectorize(squash)

        if errorf is None:
            errorf = pd_sigmoid
        self.errorf = vectorize(errorf)

    def activate(self, input_data):
        """
        Activate the layer using the specified input data.

        """
        raise NotImplementedError

    def connect(self, next_layer, contype=None):
        """
        Connect this layer to the next layer using the specified connection type. 

        Parameters
        ----------
        next_layer : Layer
            The next layer to connect to
        contype : str, optional
            The type of connection to make, by default None

        """
        raise NotImplementedError

    def error(self, expected=None):
        raise NotImplementedError

    def reset(self, weights=False):
        self.nodes = np.zeros(self.width)
        if weights and self.connected['next']:
            self.weights = np.zeros(self.width, self.connected['next'].width)
            self.errors = np.zeros(self.width, self.connected['next'].width)
            self.deltas = np.zeros(self.width, self.connected['next'].width)
            
    def __repr__(self):
        return "\n".join([f"\t{i}: ({xy[0]}) w={repr(xy[1])}" for i,xy in enumerate(zip(self.nodes.round(2), self.weights.round(2)))])

        # return " ".join([str(x) for x in zip(self.nodes.round(2), self.weights.round(2))])

    def __str__(self):
        return self.__repr__()
    

