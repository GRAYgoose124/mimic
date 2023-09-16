import os
import numpy as np
import logging


from . import *
from .utils.weights import total_norm


log = logging.getLogger(__name__)


def simple_test(M, TEST):
    for I, expected in TEST:
        actual = M.forward(I)
        print(f"Input: {I}, Expected: {expected}, Actual: {actual}")
        assert np.allclose(
            actual, expected, atol=0.05
        ), f"Failed! {M.error_fn.__class__.__name__}: {M.training_error}"


def main():
    # config
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4)
    seed = None
    if seed is not None:
        log.info(f"Setting random seed to {seed}")
        np.random.seed(seed)

    # init dataset
    log.info("Initializing dataset...")
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

    # init model
    log.info("Initializing model...")
    if not os.path.exists("data/models"):
        os.mkdir("data/models")

    if (
        not os.path.exists("data/models/xor.npz")
        or input("Load model? (Y/n): ").lower() == "n"
    ):
        # xor can be solved with only 2 hidden neurons - demo only
        M = Sequential([2, 4, 4, 1])
        can_skip_training = False
    else:
        M = Sequential.load("data/models/xor.npz")
        log.info("Loaded model from file.")
        can_skip_training = True

    # training
    skip_training = False
    if can_skip_training:
        if input("Skip training? (y/N): ").lower() == "y":
            skip_training = True

    if not skip_training:
        log.info("Training model...")

        # if can_skip_training:
        #     # normalize weights before training
        #     log.info("Normalizing weights...")
        #     total_norm(M.weights)

        T = Trainer
        T.train(
            M,
            TRAIN,
            C=TrainingConfig(epochs=10000, learning_rate=0.2),
        )

    # testing and evaluation
    log.info("Testing model...")
    R = ModelRunner(M, TEST, tests=[simple_test])
    success = R.test()

    # save model
    if success or input("Save model? (y/N): ").lower() == "y":
        log.info("Saving model...")
        M.save("data/models/xor.npz")

    # visualization
    # normalize weights before visualization, better looking
    # total_norm(M.weights)
    vis.draw_network(M, filename="xor_model.png", save=True)  # , show=True)


if __name__ == "__main__":
    main()
