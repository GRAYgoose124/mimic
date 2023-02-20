import os
import numpy as np
import logging

from mimic.models import Model
from mimic.utils.net import pd_sigmoid, draw_network
from mimic.utils.data import vary


logger = logging.getLogger()


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
        #   TODO: maybe refactor to pass (expected - actual) ?
        actual = self.evaluate(input_data, update=True)
        self.out_layer.error(expected, update=True)

        # backpropagation
        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            # Calculate this layer's error for the next("prev") layer in backpropagation.
            layer.error(update=True)

            # Calculate the deltas for this layer using the next layer's error.
            layer.deltas = (α * layer.connected['next'].errors * layer.nodes) + (momentum * layer.deltas)
            # Update weights
            layer.weights = np.array([np.subtract(x, y) for x,y in zip(layer.weights, layer.deltas)]) 

            logger.debug(f"Layer {i} update:\ndeltas:{layer.deltas}\nweights:{layer.weights}\nerrors:{layer.errors}\n")

        return self.out_layer.errors

    def train(model, dataset, epochs=1000, learning_rate=0.01, momentum=0.0, output_dir='./output'):
        # create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create subdirectory for this model
        output_dir = f"{output_dir}/{model.__class__.__name__}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Before:\n", model)
        draw_network(model, filename=f"{output_dir}/untrained.png", save=True)

        print(f"\nTraining {epochs} steps...")
        for epoch in range(epochs):
            for inp, outp in vary(dataset, variance=0.15):
                model.fit(inp, outp, α=learning_rate, momentum=momentum)

        print("After:\n", model)

        print("Testing...")
        for inp, outp in (dataset):
            res = model.evaluate(inp, update=True)

            print(f"{inp} -> {outp[0]} == {res}")
            draw_network(model, filename=f"{output_dir}/{inp}{outp}.trained.png", save=True)





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
