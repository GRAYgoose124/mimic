from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingConfig:
    learning_rate: float = 0.5
    epochs: int = 10000
    momentum: float = 0.0


@dataclass
class Dataset:
    input_data: np.ndarray
    expected_output: np.ndarray

    def split(self, test_size=0.2):
        split_index = int(len(self.input_data) * (1 - test_size))
        train_data = Dataset(
            input_data=self.input_data[:split_index],
            expected_output=self.expected_output[:split_index],
        )
        test_data = Dataset(
            input_data=self.input_data[split_index:],
            expected_output=self.expected_output[split_index:],
        )
        return train_data, test_data

    def __iter__(self):
        return iter(zip(self.input_data, self.expected_output))

    def __next__(self):
        return next(self.input_data), next(self.expected_output)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.expected_output[idx]

    def __repr__(self):
        return f"Dataset(input_data={self.input_data}, expected_output={self.expected_output})"


class ActivationFunction:
    def fn(self, x):
        pass

    def derivative(self, x):
        pass


class Sigmoid(ActivationFunction):
    def fn(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)


class ErrorFunction:
    def fn(self, expected, actual):
        pass

    def derivative(self, expected, actual):
        pass


class MSE(ErrorFunction):
    def fn(self, expected, actual):
        return np.mean(np.square(expected - actual))

    def derivative(self, expected, actual):
        return expected - actual


class MultiLayerNN:
    def __init__(
        self,
        layer_sizes,
        activation_fn: ActivationFunction = Sigmoid(),
        error_fn: ErrorFunction = MSE(),
    ):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.activations = []
        self.gradients = []

        self.activation_fn = activation_fn
        self.error_fn = error_fn

        # Initialize weights
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]))

    def forward_propagate(self, input_data):
        self.activations = [input_data]
        for weight in self.weights:
            output = self.activation_fn.fn(np.dot(self.activations[-1], weight))
            self.activations.append(output)
        return self.activations[-1]

    def backward_propagate(self, output_data, learning_rate):
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


class Trainer:
    @staticmethod
    def train(
        model: MultiLayerNN, dataset: Dataset, config: TrainingConfig = TrainingConfig()
    ):
        for epoch in range(config.epochs):
            output = model.forward_propagate(dataset.input_data)
            mse = model.backward_propagate(
                dataset.expected_output, config.learning_rate
            )

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, MSE: {mse}")

        print("Final Weights:")
        for i, w in enumerate(model.weights):
            print(f"Layer {i+1} weights:")
            print(w)

        print("Final and Expected Output:")
        for o, e in zip(output, dataset.expected_output):
            print(f"Final: {o}, Expected: {e}")
            model._trained_mse = mse


def main():
    np.set_printoptions(precision=4)

    DS = Dataset(
        input_data=np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
        expected_output=np.array([[0, 1], [1, 0], [1, 0], [0, 1]]),
    )

    M = MultiLayerNN([3, 8, 4, 2])
    Trainer.train(M, DS)

    # test
    for I, expected in DS:
        assert np.allclose(
            M.forward_propagate(I), expected, atol=0.02
        ), f"Failed! MSE: {M._trained_mse}"


if __name__ == "__main__":
    main()
