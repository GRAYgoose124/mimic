class Model:
    def __init__(self, layers: list):
        self.layers = layers

    #
    def fit(self, train_data):
        raise NotImplementedError

    def evaluate(self, input_data):
        raise NotImplementedError


class Sequential(Model):
    def __init__(self, layers: list):
        super().__init__(layers)
        # fully connected hidden
        for i, _ in enumerate(self.layers):
            if i < len(self.layers) - 1:
                self.layers[i].connect(self.layers[i + 1], contype='full')
        
        self.in_layer = self.layers[0]
        self.hidden_layers = self.layers[1:-1]
        self.out_layer = self.layers[-1]


    def evaluate(self, input_data, update=False):
        # feed-forward
        output = input_data
        for layer in self.layers:
            output = layer.activate(output, update)

        return output

    def fit(self, inp, expected, learning_rate=0.01, momentum=0.):
        # back-prop
        self.evaluate(inp, update=True)

        error = learning_rate * ((expected - self.out_layer.nodes) * self.out_layer.error(self.out_layer.nodes)) 

        for layer in self.layers[1::-1]:
            error_delta = l