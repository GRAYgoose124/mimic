from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


from .dataset import Dataset
from .models import Sequential
from .trainer import Trainer, TrainingConfig
from .utils.vis import draw_network
from .runner import ModelRunner


def simple_test(M, TEST):
    for I, expected in TEST:
        actual = M.forward(I)
        print(f"Input: {I}, Expected: {expected}, Actual: {actual}")
        assert np.allclose(
            actual, expected, atol=0.05
        ), f"Failed! {M.error_fn.__class__.__name__}: {M.training_error}"


def main():
    np.set_printoptions(precision=4)
    # for reproducibility:
    # np.random.seed(1)

    # dataset
    # TRAIN, TEST = Dataset().split(.1) # 10% separate test instead of full test and no separate train
    TRAIN = TEST = Dataset(
        input_data=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        expected_output=np.array([[0], [1], [1], [0]]),
    )

    # model
    M = Sequential([2, 4, 4, 1])

    # training
    T = Trainer
    T.train(
        M,
        TRAIN,
        C=TrainingConfig(epochs=10000, learning_rate=0.25),
    )

    # testing and evaluation
    R = ModelRunner(M, TEST)
    R.add_test(simple_test)
    R.test()

    # visualization
    draw_network(M, filename="network.png", show=True, save=True)


if __name__ == "__main__":
    main()
