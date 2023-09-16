import os
import numpy as np

from . import *


def simple_test(M, TEST):
    for I, expected in TEST:
        actual = M.forward(I)
        print(f"Input: {I}, Expected: {expected}, Actual: {actual}")
        assert np.allclose(
            actual, expected, atol=0.05
        ), f"Failed! {M.error_fn.__class__.__name__}: {M.training_error}"


def main():
    # for reproducibility:
    # np.random.seed(1)
    np.set_printoptions(precision=4)

    # dataset
    if not os.path.exists("data"):
        TRAIN = TEST = Dataset(  # TRAIN, TEST = Dataset(
            input_data=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            expected_output=np.array([[0], [1], [1], [0]]),
        )  # .split(.1)
        os.mkdir("data")
        TRAIN.save("data/ds-xor.npz")
    else:
        # TRAIN, TEST = Dataset.load("data/ds-xor.npz").split(.1)
        TRAIN = TEST = Dataset.load("data/ds-xor.npz")

    # model
    if not os.path.exists("data/models"):
        os.mkdir("data/models")

    if not os.path.exists("data/models/xor.npz"):
        M = Sequential([2, 4, 4, 1])
        can_skip_training = False
    else:
        M = Sequential.load("data/models/xor.npz")
        can_skip_training = True

    # training
    skip_training = False
    if can_skip_training:
        if input("Skip training? (y/N): ").lower() == "y":
            skip_training = True

    if not skip_training:
        T = Trainer
        T.train(
            M,
            TRAIN,
            C=TrainingConfig(epochs=10000, learning_rate=0.25),
        )

    # testing and evaluation
    R = ModelRunner(M, TEST)
    R.add_test(simple_test)
    success = R.test()

    # save model
    if success and (
        not can_skip_training or input("Save model? (y/N): ").lower() == "y"
    ):
        M.save("data/models/xor.npz")

    # visualization
    vis.draw_network(M, filename="network.png", save=True)  # , show=True)


if __name__ == "__main__":
    main()
