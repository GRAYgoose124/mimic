import numpy as np
import logging

from mimic.models.sequential import Sequential
from mimic.layers.dense import Dense

from mimic.data_utils import xor_set


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = Sequential([Dense(2),
                        Dense(4),
                        Dense(4),
                        Dense(1)])

    # model.set_logger(logger)

    model.train(xor_set, epochs=1000, learning_rate=0.01, momentum=0.01)


if __name__ == '__main__':
    main()