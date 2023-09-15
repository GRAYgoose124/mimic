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


class MultiLayerNN:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.activations = []
        self.gradients = []

        # Initialize weights
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.rand(layer_sizes[i], layer_sizes[i + 1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagate(self, input_data):
        self.activations = [input_data]
        for weight in self.weights:
            output = self.sigmoid(np.dot(self.activations[-1], weight))
            self.activations.append(output)
        return self.activations[-1]

    def backward_propagate(self, output_labels, learning_rate):
        error = output_labels - self.activations[-1]
        self.gradients = [error * self.sigmoid_derivative(self.activations[-1])]

        for i in reversed(range(len(self.activations) - 1)):
            gradient = self.gradients[0].dot(
                self.weights[i].T
            ) * self.sigmoid_derivative(self.activations[i])
            self.gradients.insert(0, gradient)

        for i in range(len(self.weights)):
            self.weights[i] += (
                self.activations[i].T.dot(self.gradients[i + 1]) * learning_rate
            )

        return np.mean(np.square(error))


class Trainer:
    @staticmethod
    def train(model, dataset: Dataset, config: TrainingConfig = TrainingConfig()):
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
    nn = MultiLayerNN([3, 8, 4, 2])
    test_ds = Dataset(
        input_data=np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
        expected_output=np.array([[0, 1], [1, 0], [1, 0], [0, 1]]),
    )

    Trainer.train(nn, test_ds)

    # test
    for test_in, exp_out in test_ds:
        assert np.allclose(
            nn.forward_propagate(test_in), exp_out, atol=0.02
        ), f"Failed! MSE: {nn._trained_mse}"


if __name__ == "__main__":
    main()
