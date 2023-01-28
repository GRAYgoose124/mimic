class Model:
    def __init__(self, layers: list):
        self.layers = layers

    def fit(self, expected, learning_rate=0.01, momentum=0.0):
        raise NotImplementedError

    def evaluate(self, input_data):
        raise NotImplementedError

    def reset(self):
        for layer in self.layers:
            layer.reset()
    
    def __repr__(self):
        return "\n".join([layer.__repr__() for i, layer in enumerate(self.layers)])
        # return "\n".join([f"{i} {layer.__repr__()}" for i, layer in enumerate(self.layers)])

    def __str__(self):
        return self.__repr__()


