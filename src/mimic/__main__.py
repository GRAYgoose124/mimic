import numpy as np
import logging

from mimic.models.sequential import Sequential
from mimic.layers.dense import Dense

from mimic.data_utils import xor_set


logger = logging.getLogger()


def main():
    model = Sequential([Dense(2),
                        Dense(4),
                        Dense(4),
                        Dense(4),
                        Dense(1)])

    model.train(xor_set, epochs=10000, learning_rate=0.076, momentum=0.01)


if __name__ == '__main__':
    main()