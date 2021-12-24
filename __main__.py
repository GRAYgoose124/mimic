import numpy as np

from models import Sequential
from layers import Dense

from utils import show_graph

if __name__ == '__main__':
    model = Sequential([Dense(2),
                        Dense(3), Dense(4),
                        Dense(1)])
    show_graph(model)
    
    xor_set = [([0.0, 1.0], 1.0),
               ([1.0, 0.0], 1.0),
               ([1.0, 1.0], 0.0),
               ([0.0, 0.0], 0.0)]

    
    
    #model.fit(xor_set)

    for inp, outp in xor_set:
        print(model.evaluate(inp, update=True))



