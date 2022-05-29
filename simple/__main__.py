import numpy as np


class Neuron:
    def __init__(self, input_neurons):
        self.in_weights = []

    def activate(self, y): # yk
        x = np.sum(np.multiply(y, self.in_weights)) # wkj
        return x * (1 - x)

    def error(self):
        delta = learning_rate * posterror * 

if __name__ == "__main__":
    pass