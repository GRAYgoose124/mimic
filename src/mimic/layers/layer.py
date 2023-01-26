import numpy as np

from numpy import vectorize

from mimic.utils import sigmoid, pd_sigmoid


class Layer:
    def __init__(self, width: int, squash: callable = None, errorf: callable = None):
        self.width = width
        self.connected = {}

        self.nodes = np.ones(width)
        self.weights = np.ones(width)
        self.errors = np.ones(width)
        self.deltas = np.zeros(width)


        self.shape = (width,)

        if squash is None:
            squash = sigmoid
        self.squash = vectorize(squash)

        if errorf is None:
            errorf = pd_sigmoid
        self.errorf = vectorize(errorf)

    def activate(self, input_data):
        raise NotImplementedError

    def connect(self, layers):
        raise NotImplementedError

    def error(self, expected=None):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.weights.round(2)}"

        # return " ".join([str(x) for x in zip(self.nodes.round(2), self.weights.round(2))])

    def __str__(self):
        return self.__repr__()
    

