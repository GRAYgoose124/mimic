import numpy as np
import logging

from mimic.models.sequential import Sequential
from mimic.layers.dense import Dense

from mimic.net_utils import show_network
from mimic.data_utils import xor_set, vary


logger = logging.getLogger()


def train(model, dataset, steps=1000):
    print("Before:\n", model)

    print(f"\nTraining {steps} steps...")
    for epoch in range(steps):
        for inp, outp in vary(dataset):
            model.backprop(inp, outp)

    print("After:\n", model)

    print("Testing...")
    for inp, outp in vary(dataset):
        res = model.evaluate(inp)
        print(f"{inp} -> {outp[0]} == {res}")

    show_network(model)


def main():
    model = Sequential([Dense(2),
                        Dense(4),
                        Dense(4),
                        Dense(4),
                        Dense(1)])

    # 2 inputs, 1 output
    train(model, xor_set, steps=1000)


if __name__ == '__main__':
    main()