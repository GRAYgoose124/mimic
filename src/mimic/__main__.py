from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


from .dataset import Dataset
from .models.basic import MultiLayerNN
from .trainer import Trainer, TrainingConfig
from .utils.net import draw_network


class ModelRunner:
    def __init__(self, M, DS):
        self.M = M
        self.DS = DS

        self._tests = []

    @property
    def tests(self):
        return self._tests

    def add_test(self, test):
        self._tests.append(test)

    def test(self):
        try:
            for test in self.tests:
                test(self.M, self.DS)

            return True
        except AssertionError as e:
            print(e)
            return False


def simple_test(M, TEST):
    for I, expected in TEST:
        assert np.allclose(
            M.forward_propagate(I), expected, atol=0.02
        ), f"Failed! MSE: {M._trained_mse}"


def main():
    np.set_printoptions(precision=4)

    # for reproducibility:
    # np.random.seed(1)

    # dataset
    # TRAIN, TEST = Dataset().split(.1) # 10% separate test instead of full test and no separate train
    TRAIN = TEST = Dataset(
        input_data=np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
        expected_output=np.array([[0, 1], [1, 0], [1, 0], [0, 1]]),
    )

    # model
    M = MultiLayerNN([3, 8, 4, 2])

    # training
    T = Trainer
    T.train(M, TRAIN, config=TrainingConfig(epochs=10000, learning_rate=0.35))

    # testing and evaluation
    R = ModelRunner(M, TEST)
    R.add_test(simple_test)
    if R.test():
        print(f"Passed! MSE: {M._trained_mse}")

    # visualization
    draw_network(M, filename="network.png", show=True, save=True)


if __name__ == "__main__":
    main()
