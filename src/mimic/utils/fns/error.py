import numpy as np

from .base import DifferentFn


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
