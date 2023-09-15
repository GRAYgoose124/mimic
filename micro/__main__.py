import numpy as np


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


# Initialize input and output
input_data = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

output_labels = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])

# Create neural network instance with 3 layers: 3 neurons in input, 4 in hidden, and 2 in output
nn = MultiLayerNN([3, 4, 2])

# Learning rate and epochs
learning_rate = 0.25
epochs = 10000

# Training loop
for epoch in range(epochs):
    output = nn.forward_propagate(input_data)
    mse = nn.backward_propagate(output_labels, learning_rate)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, MSE: {mse}")

print("Final Weights:")
for i, w in enumerate(nn.weights):
    print(f"Layer {i+1} weights:")
    print(w)

print(f"Final Output: {output}, Expected: {output_labels}")
print(f"Final MSE: {mse}")

# test
test_data = np.array([[1, 1, 0]])
print(f"Test data: {test_data}")
print(f"Test output: {nn.forward_propagate(test_data)}, Expected: {np.array([[0, 1]])}")
