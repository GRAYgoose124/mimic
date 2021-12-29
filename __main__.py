import numpy as np
import logging

from models import Sequential
from layers import Dense

from utils import show_graph

logger = logging.getLogger()


if __name__ == '__main__':
    model = Sequential([Dense(2),
                        Dense(2),
                        Dense(2),
                        Dense(1)])

    # show_graph(model)
    
    xor_set = [([0.0, 1.0], 1.0),
               ([1.0, 0.0], 1.0),
               ([1.0, 1.0], 0.0),
               ([0.0, 0.0], 0.0)]
    
    print("untrained\n")
    print(model)
    for inp, outp in xor_set:
        print(model.evaluate(inp))

    te = 0.0
    for epoch in range(1):
        for inp, outp in xor_set:
            model.evaluate(inp, update=True)
            te = model.fit(outp)
        # if epoch % 100:
        print("error: ", te)

    # model.reset()

    print("trained\n")
    print(model)

    for inp, outp in xor_set:
        print(model.evaluate(inp))

