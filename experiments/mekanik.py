import numpy as np
from numba import njit


@njit(fastmath=True)
def train(layers, X, y, learning_rate, epochs):
    for _ in range(epochs):
        for i in range(X.shape[0]):
            activations = [X[i : i + 1]]
            for layer in layers:
                activations.append(np.tanh(np.dot(activations[-1], layer[0]) + layer[1]))
                
            delta = activations[-1] - y[i : i + 1]
            for layer_index in range(len(layers) - 1, -1, -1):
                W, b = layers[layer_index]
                layers[layer_index] = (
                    W - learning_rate * np.dot(activations[layer_index].T, delta), 
                    b - learning_rate * np.sum(delta, axis=0)
                )
                delta = np.dot(delta, W.T)


@njit(fastmath=True)
def predict(layers, X):
    X = np.ascontiguousarray(X)

    activation = X
    for layer in layers:
        activation = np.tanh(np.dot(activation, layer[0]) + layer[1])
    return activation


class KAN:
    def __init__(self, layer_sizes):
        self.layers = [
            (np.random.randn(layer_sizes[i], layer_sizes[i + 1]), 
             np.random.randn(layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)
        ]

    def train(self, X, y, learning_rate, epochs):
        return train(self.layers, X, y, learning_rate, epochs)

    def predict(self, X):
        return predict(self.layers, X)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(42)
    X = np.random.uniform(0, 1, (1000, 1))
    y = 0.5 * np.sin(4 * np.pi * X) + 0.5 + 0.1 * np.random.randn(1000, 1)

    kan = KAN([1, 20, 20, 1])
    kan.train(X, y, learning_rate=0.01, epochs=1000)

    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_pred = kan.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label="Training data")
    plt.plot(X_test, y_pred, "r-", label="Predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Neural Network Fitting Uniform Data")
    plt.legend()
    plt.show()
