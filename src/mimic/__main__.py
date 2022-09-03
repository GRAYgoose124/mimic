import numpy as np
import logging

from mimic.models.sequential import Sequential
from mimic.layers.dense import Dense

from mimic.utils import show_graph


logger = logging.getLogger()


def main():
    model = Sequential([Dense(2),
                    Dense(4),
                    Dense(3),
                    Dense(1)])


    xor_set = [([0.0, 1.0], [1.0]),
               ([1.0, 0.0], [1.0]),
               ([1.0, 1.0], [0.0]),
               ([0.0, 0.0], [0.0])]
    
    print(model)

    for epoch in range(10000):
        for inp, outp in xor_set:
            model.backprop(inp, outp)

    
    print("\ntrained: ", model)
    show_graph(model)

    for inp, outp in xor_set:
        res = model.evaluate(inp)
        print(f"{inp} -> {outp[0]} == {res}")



if __name__ == '__main__':
    main()