import numpy as np

from .base import DifferentFn


class ActivationFunction(DifferentFn):
    def fn(self, x):
        pass

    def derivative(self, x):
        pass


class Sigmoid(ActivationFunction):
    def fn(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)
