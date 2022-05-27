from utils import pd_sigmoid
import numpy as np

class Model:
    def __init__(self, layers: list):
        self.layers = layers

    def fit(self, expected, learning_rate=0.01, momentum=0.0):
        raise NotImplementedError

    def evaluate(self, input_data):
        raise NotImplementedError

    
    def __repr__(self):
        return "\n".join([layer.__repr__() for i, layer in enumerate(self.layers)])
        # return "\n".join([f"{i} {layer.__repr__()}" for i, layer in enumerate(self.layers)])

    def __str__(self):
        return self.__repr__()


class Sequential(Model):
    def __init__(self, layers, conntype='random'):
        super().__init__(layers)

        self.layers = layers
        
        # fully connected hidden
        self.layers[0].connect(self.layers[1], 'ones')
        for i, _ in enumerate(self.layers[1:-1]):
            self.layers[i + 1].connect(self.layers[i + 2], conntype)

        self.layers[-1].connected_layers['prev'] = self.layers[-2]

        self.in_layer = self.layers[0]
        self.hidden_layers = self.layers[1:-1]
        self.out_layer = self.layers[-1]

    def evaluate(self, input_data, update=False):
        # feed-forward
        output = input_data
        for layer in self.layers:
            output = layer.activate(output, update)

        return output

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def backprop(self, input_data, expected, learning_rate=0.1, momentum=0.0):
        self.evaluate(input_data, update=True)

        self.out_layer.errors = (expected - self.out_layer.nodes) * self.out_layer.nodes * (1 - self.out_layer.nodes) # pd_sigmoid
        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            nl = layer.connected_layers['next']
  
            delta = (learning_rate * np.multiply(nl.errors, layer.nodes)) + (momentum * layer.errors)

            self.hidden_layers[i].weights = np.subtract(layer.weights, delta)
            self.hidden_layers[i].errors = (nl.errors * layer.weights) * layer.nodes * (1 - layer.nodes) # pd_sigmoid

        # for i, layer in reversed(list(enumerate(self.hidden_layers))):
        #     layer.errors = np.dot(layer.connected_layers['next'].errors, layer.weights.transpose())
        #     gradient = np.matmul(layer.errors, layer.error(layer.nodes))
        #     self.hidden_layers[i].weights -= learning_rate * gradient


if __name__ == '__main__':
    from layers import Dense
    net = [Dense(1), Dense(2), Dense(3), Dense(4)]
    model = Sequential(net, 'full')

    print(model)
    print(len(model.layers))

    for layer in model.layers:
        try:
            print('l')
            print(id(layer.connected_layers['prev']))
            print(id(layer))
            print(id(layer.connected_layers['next']))
        except KeyError:
            pass


    print(model.layers[0].__dict__)
    print(model.layers[1].__dict__)
    print(model.layers[2].__dict__)
    print('3', model.layers[3].__dict__)
