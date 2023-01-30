import os
import numpy as np

from mimic.models import Model
from mimic.net_utils import pd_sigmoid, draw_network
from mimic.data_utils import vary


class Sequential(Model):
    """ Sequential model class.
    
    The layers in the sequential model are linearly connected.

    The first layer is the input layer, the last layer is the output layer. The
    layers in between are the hidden layers. Only the hidden layers are trained.

    The model uses back-propagation to train the hidden layers with the `fit` method. 
    `evaluate` is used to evaluate the model on a given input.

    """
    def __init__(self, layers, conntype='random'):
        super().__init__(layers)
        self.connect(conntype=conntype)
    
    def connect(self, conntype='random'):
        # connect the first hidden layer to the input layer
        self.layers[0].connect(self.layers[1], 'ones')

        # fully connected hidden layers
        for i, _ in enumerate(self.layers[1:-1]):
            self.layers[i + 1].connect(self.layers[i + 2], conntype)

        # connect the output layer to the last hidden layer
        # TODO: maybe refactor to use connect method? Uncertain if this is properly feeding data forward.
        self.layers[-2].connect(self.layers[-1], 'ones')

        self.in_layer = self.layers[0]
        self.hidden_layers = self.layers[1:-1]
        self.out_layer = self.layers[-1]

    def evaluate(self, input_data, update=False):
        output = input_data
        for layer in self.layers:
            output = layer.activate(output, update)

        return output

    def fit(self, input_data, expected, α=0.01, momentum=0.0):
        # backpropagation
        #   TODO: maybe refactor to pass (expected - actual) ?
        actual = self.evaluate(input_data, update=True)
        self.out_layer.error(expected)

        # update weights
        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            layer.deltas = (α * layer.connected['next'].errors * layer.nodes) + (momentum * layer.deltas)
            # Calculate this layer's error for the next layer in backpropagation.
            layer.error(update=True)

            
            new_weights = np.array([np.subtract(x, y) for x,y in zip(layer.weights, layer.deltas)]) 
            layer.weights = new_weights

        return self.out_layer.errors

    def train(self, dataset, epochs=1000, learning_rate=0.01, momentum=0.0, output_dir='./output'):
        # create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create subdirectory for this model
        output_dir = f"{output_dir}/{self.__class__.__name__}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Before:\n", self)
        draw_network(self, filename=f"{output_dir}/untrained.png", save=True)

        print(f"\nTraining {epochs} steps...")
        for epoch in range(epochs):
            for inp, outp in vary(dataset):
                self.fit(inp, outp, α=learning_rate, momentum=momentum)

        print("After:\n", self)

        print("Testing...")
        for inp, outp in (dataset):
            res = self.evaluate(inp, update=True)

            print(f"{inp} -> {outp[0]} == {res}")
            draw_network(self, filename=f"{output_dir}/{inp}{outp}.trained.png", save=True)


if __name__ == '__main__':
    from layers import Dense
    net = [Dense(1), Dense(2), Dense(3), Dense(4)]
    model = Sequential(net, 'full')

    print(model)
    print(len(model.layers))

    for layer in model.layers:
        try:
            print('l')
            print(id(layer.connected['prev']))
            print(id(layer))
            print(id(layer.connected['next']))
        except KeyError:
            pass

    print(model.layers[0].__dict__)
    print(model.layers[1].__dict__)
    print(model.layers[2].__dict__)
    print('3', model.layers[3].__dict__)
