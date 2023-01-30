class Model:
    """
    A general model class that can be used to create different types of models.

    The `fit` and `evaluate` methods are not implemented here, but in the
    subclasses which inherit from this class. This is because how layers are
    composed can effect the training technique.

    """
    def __init__(self, layers: list):
        self.layers = layers

    def connect(self, conntype=''):
        raise NotImplementedError

    def train(self, dataset, epochs=1000, learning_rate=0.01, momentum=0.0, output_dir='/output'):
        pass

    def fit(self, input_data, expected, learning_rate=0.01, momentum=0.0):
        """
        A general fit method that can be used to train the model.

        This is one step of the training process.
        """
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


