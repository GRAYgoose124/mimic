import numpy as np

from mimic.models.model import Model
from mimic.utils import pd_sigmoid


class Sequential(Model):
    def __init__(self, layers, conntype='random'):
        super().__init__(layers)

        self.layers = layers
        
        # fully connected hidden
        self.layers[0].connect(self.layers[1], 'ones')
        for i, _ in enumerate(self.layers[1:-1]):
            self.layers[i + 1].connect(self.layers[i + 2], conntype)

        self.layers[-1].connected['prev'] = self.layers[-2]

        self.in_layer = self.layers[0]
        self.hidden_layers = self.layers[1:-1]
        self.out_layer = self.layers[-1]

    def evaluate(self, input_data, update=False):
        output = input_data
        for layer in self.layers:
            output = layer.activate(output, update)

        return output

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def backprop(self, input_data, expected, learning_rate=0.01, momentum=0.0):
        actual = self.evaluate(input_data, update=True)

        self.out_layer.error(expected)
        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            nl = layer.connected['next']
  
            delta = (learning_rate * nl.errors * layer.nodes) + (momentum * layer.deltas)
            self.hidden_layers[i].deltas = delta

            new_weights = np.array([np.subtract(x, y) for x,y in zip(layer.weights, delta)]) 
            self.hidden_layers[i].weights = new_weights # hacked, should just  be able to use -= delta:/
            self.hidden_layers[i].errors = layer.error()  # pd_sigmoid


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
