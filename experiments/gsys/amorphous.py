import numpy as np


class AmorphousLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        self.last_input = x
        self.last_output = np.tanh(np.dot(x, self.weights) + self.bias)
        return self.last_output

    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T) * (1 - np.power(self.last_output, 2))
        grad_weights = np.dot(self.last_input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        return grad_input, grad_weights, grad_bias

class AmorphousNetwork:
    def __init__(self, g_system):
        self.g_system = g_system
        self.graph = g_system.graph
        self.node_order = list(self.graph.nodes())
        self.layers = {node: AmorphousLayer(1, 1) for node in self.node_order}
        self.input_nodes = [node for node in self.node_order if self.graph.in_degree(node) == 0]
        self.output_nodes = [node for node in self.node_order if self.graph.out_degree(node) == 0]

    def forward(self, x):
        node_outputs = {node: np.zeros_like(x) for node in self.node_order}
        for input_node in self.input_nodes:
            node_outputs[input_node] = x

        for node in self.node_order:
            if node not in self.input_nodes:
                inputs = [node_outputs[pred] for pred in self.graph.predecessors(node)]
                if inputs:
                    node_input = np.mean(inputs, axis=0)
                    node_outputs[node] = self.layers[node].forward(node_input)

        return np.mean([node_outputs[node] for node in self.output_nodes], axis=0)

    def backward(self, x, grad_output):
        node_outputs = {node: np.zeros_like(x) for node in self.node_order}
        for input_node in self.input_nodes:
            node_outputs[input_node] = x

        for node in self.node_order:
            if node not in self.input_nodes:
                inputs = [node_outputs[pred] for pred in self.graph.predecessors(node)]
                if inputs:
                    node_input = np.mean(inputs, axis=0)
                    node_outputs[node] = self.layers[node].forward(node_input)

        node_grads = {node: np.zeros_like(x) for node in self.node_order}
        for output_node in self.output_nodes:
            node_grads[output_node] = grad_output / len(self.output_nodes)

        for node in reversed(self.node_order):
            if node not in self.input_nodes:
                grad_input, grad_weights, grad_bias = self.layers[node].backward(node_grads[node])
                self.layers[node].weights -= self.learning_rate * grad_weights
                self.layers[node].bias -= self.learning_rate * grad_bias
                for pred in self.graph.predecessors(node):
                    node_grads[pred] += grad_input / self.graph.out_degree(pred)

    def train(self, X, y, learning_rate=0.01, epochs=100):
        interval = epochs // 10
        self.learning_rate = learning_rate
        for i in range(epochs):
            self.backward(X, error := self.forward(X) - y)  
            if i % interval == 0:
                print(f"Epoch {i}, Loss: {np.mean(error ** 2)}")
