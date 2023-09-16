import os
import numpy as np
from tensorflow import keras


if __name__ == "__main__":
    # Load and preprocess the MNIST dataset
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = keras.datasets.mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)

    from mimic.utils.dataset import Dataset

    # to np arrays
    images = np.concatenate((train_images, test_images))
    labels = np.concatenate((train_labels, test_labels))

    dataset = Dataset(
        input_data=images,
        expected_output=labels,
    )

    dataset.save("data/mnist.npz")
