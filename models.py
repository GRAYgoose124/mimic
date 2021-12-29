from utils import pd_sigmoid
import numpy as np

class Model:
    def __init__(self, layers: list):
        self.layers = layers

    #
    def fit(self, train_data):
        raise NotImplementedError

    def evaluate(self, input_data):
        raise NotImplementedError

    def __repr__(self):
        return "\n".join([layer.__repr__() for layer in self.layers])

    def __str__(self):
        return self.__repr__()


class Sequential(Model):
    def __init__(self, layers):
        super().__init__(layers)

        self.layers = layers
        # fully connected hidden
        for i, _ in enumerate(self.layers):
            if i < len(self.layers) - 1:
                self.layers[i].connect(self.layers[i + 1], 'random')
        
  

    def evaluate(self, input_data, update=False):
        # feed-forward
        output = input_data
        for layer in self.layers:
            output = layer.activate(output, update)

        return output

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def fit(self, expected, learning_rate=0.01, momentum=0.):
        self.out_layer.errors = np.array(self.out_layer.nodes - expected)

        for layer in self.layers[1::-1]:
            print(layer.weights, layer.connected_layers['next'].errors)

            errors = layer.connected_layers['next'].errors.dot(layer.weights)
            print(errors)

            errors = errors * layer.error(layer.nodes)
            update = learning_rate * errors * layer.connected_layers['next'].nodes
            layer.errors = errors
            layer.weights -= update

        total_error = 0.0
        for layer in self.layers:
            total_error += layer.errors
        return total_error


if __name__ == '__main__':
    from layers import Dense
    net = [Dense(2), Dense(3), Dense(3), Dense(4)]
    model = Sequential(net)

    print(model)
    print(len(model.layers))
    for layer in model.layers[1:-1]:
        print(id(layer.connected_layers['prev']), id(layer), id(layer.connected_layers['next']))

    
