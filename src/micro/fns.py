from abc import ABC, abstractmethod

import numpy as np


class DifferentFn(ABC):
    @abstractmethod
    def fn(self):
        pass

    @abstractmethod
    def derivative(self):
        pass


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


class ErrorFunction(DifferentFn):
    def fn(self, expected, actual):
        pass

    def derivative(self, expected, actual):
        pass


class MSE(ErrorFunction):
    def fn(self, expected, actual):
        return np.mean(np.square(expected - actual))

    def derivative(self, expected, actual):
        return expected - actual


class CategoryCrossEntropy(ErrorFunction):
    def fn(self, expected, actual):
        return -np.sum(expected * np.log(actual))

    def derivative(self, expected, actual):
        return expected / actual
