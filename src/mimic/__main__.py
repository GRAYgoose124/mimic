import numpy as np
import logging

from mimic.models.sequential import Sequential
from mimic.layers.dense import Dense
from mimic.optimizers.simpleton import Simpleton

from mimic.utils.data import xor_set


def main():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    #filter out matplotlib loggers
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    #filter out PIL loggers
    logging.getLogger('PIL').setLevel(logging.INFO)
    
    model = Sequential([Dense(2),
                        Dense(4),
                        Dense(4),
                        Dense(1)])

    # model.set_logger(logger)

    model.train(xor_set, epochs=10, learning_rate=0.5, momentum=0.01)


if __name__ == '__main__':
    main()