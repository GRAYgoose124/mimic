import numpy as np

from .base import ANN
from ..fns import ActivationFunction, ErrorFunction, MSE, Sigmoid


class Sequential(ANN):
    def __init__(
        self,
        layer_sizes,
        activation_fn: ActivationFunction = Sigmoid(),
        error_fn: ErrorFunction = MSE(),
    ):
        super().__init__(layer_sizes, activation_fn, error_fn)

    def forward(self, input_data):
        self.activations = [input_data]
        for weight in self.weights:
            output = self.activation_fn.fn(np.dot(self.activations[-1], weight))
            self.activations.append(output)
        return self.activations[-1]

    def backward(self, output_data, learning_rate):
        self.gradients = [
            self.error_fn.derivative(output_data, self.activations[-1])
            * self.activation_fn.derivative(self.activations[-1])
        ]

        for i in reversed(range(len(self.activations) - 1)):
            gradient = self.gradients[0].dot(
                self.weights[i].T
            ) * self.activation_fn.derivative(self.activations[i])
            self.gradients.insert(0, gradient)

        for i in range(len(self.weights)):
            self.weights[i] += (
                self.activations[i].T.dot(self.gradients[i + 1]) * learning_rate
            )

        return self.error_fn.fn(output_data, self.activations[-1])
