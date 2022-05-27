import numpy as np
from math import e


def sigmoid(weighted_sum, gamma=1):
    return 1 / (1 + e ** (-gamma * weighted_sum))


def pd_sigmoid(output):
    return output * (1 - output)


squashing_functions = {'sigmoid': np.vectorize(sigmoid)}
error_functions = {'pd_sigmoid': np.vectorize(pd_sigmoid)}


class Model:
    def __init__(self, layers: list):
        self.layers = layers

    def plot(self):
        raise NotImplemented

    #
    def fit(self, train_data):
        raise NotImplemented

    def evaluate(self, test_data):
        raise NotImplemented

    def predict(self, new_data):
        raise NotImplemented

    #
    def set_dropout(self):
        raise NotImplemented


class Recurrent(Model):
    pass


class Sequential(Model):
    def __init__(self, layers: list):
        super().__init__(layers)

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                self.layers[i].connect(self.layers[i + 1])

    def feed_forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.activate(output)
        return output

    def back_prop(self):
        for layer in self.layers[-1]:
            error = layer.calc_error()


class Layer:
    def __init__(self, width: int, squash_func: str, error_func: str):
        self.width = width
        self.nodes = np.zeros(width)
        self.weights = None
        self.squash = squashing_functions[squash_func]
        self.calc_error = error_func[error_func]
        self.connected_layers = {}

    def activate(self, input_data):
        raise NotImplemented

    def connect(self, layers):
        raise NotImplemented

    def __repr__(self):
        print_string = f'L {self.nodes} |\nW {self.weights} ->\n\n'
        return "%02.f" % print_string.translate({ord(c): None for c in '[],'})


class Dense(Layer):
    def __init__(self, width: int, squash_func='sigmoid', error_func='pd_sigmoid'):
        super().__init__(width, squash_func, error_func)

    def activate(self, input_data):
        if 'prev' not in self.connected_layers:
            self.nodes = input_data
        else:
            self.nodes = self.squash(input_data.dot(self.connected_layers['prev'].weights))
        return self.nodes

    def connect(self, next_layer):
        self.weights = np.random.random_sample((self.width, next_layer.width))
        self.connected_layers['next'] = next_layer
        next_layer.connected_layers['prev'] = self


if __name__ == '__main__':
    model = Sequential([Dense(2),
                        Dense(3), Dense(4),
                        Dense(1)])

    xor_set = np.array([([0.0, 1.0], 1.0),
                        ([1.0, 0.0], 1.0),
                        ([1.0, 1.0], 0.0),
                        ([0.0, 0.0], 0.0)])

    # model.fit(xor_set)
    # model.evaluate(xor_set)
    # model.predict([1.0, 1.0])

    print(model.layers)

    model.feed_forward(np.array([1.0, 0.0]))

    print(model.layers)
