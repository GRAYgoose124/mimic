from math import e
from random import random

class SimpleNet:
    """ This simple net is based on the notes by J.G Makin found here:
            https://inst.eecs.berkeley.edu/~cs182/sp06/notes/backprop.pdf

        1. ∆w_kj = -α * ∂E/∂w_kj
            is the weight change for a weight connecting a neuron in layer k to a neuron in layer j.

        2. ∂E/∂w_kj = ∂E/∂y_j * ∂y_j/∂x_j * ∂x_j/∂w_kj
            using the chain rule to expand this partial derivative.

            3. x_j = Σ_k∈Ki(w_kj * y_k)
                is the weighted sum of inputs(y_k) into the j-th node :=
                    ∂x_j/∂w_kj = y_k
                        is the partial derivative of the weighted sum to the weights connecting neuron j to neuron k.
                        y_k are the inputs into the y-th node of the k-th layer

            4. y_j = f(x_j) = 1 / (1 + e^-(γ*x_j)
                is the sigmoid function :=
                    a. ∂y_j/∂x_j = y_j * (1 - y_j)
                        is the partial derivative of the sigmoid function to it's weighted sum.

                    b. δ_j = -∂E/∂y_j * ∂y_j/∂x_j
                        is the error term. it is the partial derivative of the weighted sum to the output
                        * the partial error to the output of the y-th node of the j-th layer.

            if j is the output layer:
                5. ∂E/∂y_j = -(t_j - y_j)

                6. ∂E/∂w_kj = (-(t_j - y_j)) * (y_j * (1 - y_j)) * (y_k)
                    is a weight change of the output layer by combining the previous steps.

            if j is a hidden layer:
                5. ∂E/∂y_j = Σ_i∈Ij(∂E/∂y_i * ∂y_i/∂x_i * ∂x_i/∂y_j)
                    the error caused by y_j propagating into the activations of the next(i-th) layer.

                6. ∂x_i/∂y_j = w_ji :=
                    ∂E/∂y_j = -Σ_i∈Ij( δ_i * w_ji)

                7. ∂E/∂w_kj = (-Σ_i∈Ij(δ_i * w_ji)) * (y_j * (1 - y_j)) * (y_k)
                    is a weight change in hidden layer i by combining the previous steps.

            finally:
                8. ∂E/∂w_kj = -δ_j*y_k
                    despite the differences between output and hidden layers.
                        where y_k is the activation of the neuron before it's weighting

                9. ∆w_kj(n) = (αδ_j * y_k) + (η * ∆w_kj(n - 1))
              w      from combining 1 and 8
                        where α == learning rate from [0, 1]
                              η == momentum from [0, 1]
                              δ_j == error term associated with a neuron after it's weight.

                if j is output layer
                    10. δ_j = (t_j - y_j) * y_j * (1 - y_j)

                if j is a hidden layer
                    10. δ_j = Σ_i∈Ij(δ_i * w_ji) * y_j * (1 - y_j)

    """
    def __init__(self, input_size, hidden_size, n_hidden_layers, output_size):
        self.size = input_size, (hidden_size, n_hidden_layers), output_size

        self.input_layer = SimpleLayer(input_size)
        self.hidden_layers = [SimpleLayer(hidden_size) for _ in range(n_hidden_layers)]
        self.output_layer = SimpleLayer(output_size)

        self.final_error = 0.0

        # connect input layer to first hidden layer
        for n in self.input_layer:
            n.set_weights([random() for _ in range(self.size[1][0])])

        # connect all hidden layers except the last
        for h in self.hidden_layers[:-1]:
            for n in h:
                n.set_weights([random() for _ in range(self.size[1][0])])

        # connect the last hidden layer to the output layer
        for n in self.hidden_layers[-1]:
            n.set_weights([random() for _ in range(self.size[2])])

    def backprop(self):
        pass

    def feedforward(self, input_data):
        if len(input_data) != self.size[0]:
            return

        for n in self.input_layer.neurons:
            n.activate()

    def train(self):
        pass

    # def output_error(self, target):
    #     """ This error only applies to the j_th layer when j is the output layer.
    #         This is the only layer on which error is defined.
    #
    #         E := 1/2 * Σ_1→j((t_j - y_j)^2)
    #     """
    #     error_sum = 0.0
    #     for t, n in target, self.output_layer:
    #         error_sum = (t - n)**2
    #
    #     error = 0.5 * error_sum
    #     return error

    def __repr__(self):
        return f'input:\n{self.input_layer}\nhidden:\n{self.hidden_layers}\noutput:\n{self.output_layer}'


class SimpleLayer:
    def __init__(self, size):
        self.neurons = [SimpleNeuron(i) for i in range(size)]

    def __iter__(self):
        return iter(self.neurons)

    def __repr__(self):
        return f'\t{self.neurons}\n'


class SimpleNeuron:
    def __init__(self, index):
        self.index = index
        self.output = 0.0
        self.weighted_sum = 0.0
        self.output_weights = []

    def set_weights(self, weights):
        self.output_weights = weights

    def activate(self, data, weights):
        """ The neuron activation function.

            x_j = Σ_k∈Ki(w_kj * y_k)
                from 3

            y_j = f(x_j) = sigmoid(x_j)
                from 4
        """
        # compute x_j.
        self.weighted_sum = 0
        for d, w in data, weights:
            self.weighted_sum += d*w

        # Here we squash the weighted sum with the sigmoid function to get y_j.
        self.output = self.sigmoid(self.weighted_sum)
        return self.output

    def sigmoid(self, weighted_sum, gamma=1):
        """ Sigmoid activation function used for squashing the input of a neuron into an
            output range of -1.0 to 1.0.

            y_j = f(x_j) = 1 / (1 + e^-(γ*x_j)
                from 4
        """
        return 1 / (1 + e**(-gamma*weighted_sum))

    def pd_sigmoid(self):
        """ Partial derivative of the sigmoid function used in the error calculation step.

            ∂y_j/∂x_j = ∂f(x_j)/∂x_j = y_j * (1 - y_j)
                from 4a
        """
        return self.output * (1 - self.output)

    def __repr__(self):
        return f'{self.output:.3f} {self.error:.3f} {[round(n, 3) for n in self.output_weights]}'


if __name__ == '__main__':
    xor_set = [([0.0, 1.0], 1.0),
               ([1.0, 0.0], 1.0),
               ([1.0, 1.0], 0.0),
               ([0.0, 0.0], 0.0)]

    net = SimpleNet(2, 2, 1, 1)
    print(net)
