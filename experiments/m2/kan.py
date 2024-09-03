import numpy as np
from numba import njit

from .utils import FileMixin


@njit(fastmath=True)
def create_layer(input_size, output_size):
    return (
        np.random.randn(input_size, output_size),  # weights
        np.random.randn(output_size),  # bias
    )


@njit(fastmath=True, parallel=True)
def layer_forward(layer, X):
    return np.tanh(np.dot(X, layer[0]) + layer[1])


@njit(fastmath=True, parallel=True)
def layer_backward(layer, X, delta, learning_rate):
    weights, bias = layer
    d_weights = np.dot(X.T, delta)
    d_bias = np.sum(delta, axis=0)

    new_weights = weights - learning_rate * d_weights
    new_bias = bias - learning_rate * d_bias

    delta_prev = np.dot(delta, weights.T)

    return (new_weights, new_bias), delta_prev


@njit(fastmath=True)
def create_kan(layer_sizes):
    return [
        create_layer(layer_sizes[i], layer_sizes[i + 1])
        for i in range(len(layer_sizes) - 1)
    ]


@njit(fastmath=True)
def train_batch(layers, X_batch, y_batch, learning_rate):
    batch_size = X_batch.shape[0]
    activations = [X_batch]
    for layer in layers:
        activations.append(layer_forward(layer, activations[-1]))

    delta = activations[-1] - y_batch
    new_layers = []
    for layer_index in range(len(layers) - 1, -1, -1):
        layer = layers[layer_index]
        new_layer, delta = layer_backward(
            layer, activations[layer_index], delta, learning_rate
        )
        new_layers.insert(0, new_layer)

    return new_layers


@njit(fastmath=True)
def train(layers, X, y, learning_rate, epochs, batch_size):
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size

    for _ in range(epochs):
        # Shuffle the data
        permutation = np.random.permutation(n_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            layers = train_batch(layers, X_batch, y_batch, learning_rate)

    return layers


@njit(fastmath=True)
def predict(layers, X):
    X = np.ascontiguousarray(X)

    activation = X
    for layer in layers:
        activation = layer_forward(layer, activation)

    return activation



class KAN(FileMixin):
    def __init__(self, layer_sizes=None):
        if layer_sizes is not None:
            self.layers = create_kan(layer_sizes)
            self.layer_sizes = layer_sizes
        else:
            self.layers = None
            self.layer_sizes = None

    def train(self, X, y, learning_rate, epochs, batch_size):
        if not self.layers:
            raise ValueError("Layers are not initialized.")

        # find any checkpoints and load them
        checkpoint = self.get_latest_checkpoint(".")
        if checkpoint:
            print(f"Loading checkpoint: {checkpoint}")
            self.layers = self.load(checkpoint)
            self.layer_sizes = [layer[0].shape[0] for layer in self.layers]
            if input("Skip training? (y/n): ") == "y":
                return

        self.layers = train(self.layers, X, y, learning_rate, epochs, batch_size)

        self.save_checkpoint(f"{self.__class__.__name__}_{id(self)}")

    def predict(self, X):
        return predict(self.layers, X)