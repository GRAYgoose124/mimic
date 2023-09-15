from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from .dataset import Dataset
from .models.basic import MultiLayerNN
from .trainer import Trainer


def main():
    np.set_printoptions(precision=4)

    DS = Dataset(
        input_data=np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
        expected_output=np.array([[0, 1], [1, 0], [1, 0], [0, 1]]),
    )

    M = MultiLayerNN([3, 8, 4, 2])
    Trainer.train(M, DS)

    # test
    for I, expected in DS:
        assert np.allclose(
            M.forward_propagate(I), expected, atol=0.02
        ), f"Failed! MSE: {M._trained_mse}"


if __name__ == "__main__":
    main()
