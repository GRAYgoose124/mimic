import numpy as np
import logging

from models import Sequential
from layers import Dense

from utils import show_graph


logger = logging.getLogger()


if __name__ == '__main__':
    model = Sequential([Dense(2),
                        Dense(4),
                        Dense(4),
                        Dense(1)])

    
    xor_set = [([0.0, 1.0], [1.0]),
               ([1.0, 0.0], [1.0]),
               ([1.0, 1.0], [0.0]),
               ([0.0, 0.0], [0.0])]
    

    for epoch in range(500):
        for inp, outp in xor_set:
            model.backprop(inp, outp)


    for inp, outp in xor_set:
        res = model.evaluate(inp)
        print(f"{inp} -> {outp[0]} == {res}")