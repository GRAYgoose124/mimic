from abc import ABC, abstractmethod

import numpy as np


class Derivable(ABC):
    @abstractmethod
    def fn(self):
        pass

    @abstractmethod
    def derivative(self):
        pass


class ActivationFunction(Derivable):
    def fn(self, x):
        pass

    def derivative(self, x):
        pass


class Sigmoid(ActivationFunction):
    def fn(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class ErrorFunction(Derivable):
    def fn(self, expected, actual):
        pass

    def derivative(self, expected, actual):
        pass


class MSE(ErrorFunction):
    def fn(self, expected, actual):
        return np.mean(np.square(expected - actual))

    def derivative(self, expected, actual):
        return expected - actual
