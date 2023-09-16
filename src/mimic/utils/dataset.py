import os
import numpy as np

from dataclasses import dataclass


def rand_dist(value, error=0.33):
    return value + np.random.uniform(-error, error, value.shape)


@dataclass
class Dataset:
    input_data: np.ndarray
    expected_output: np.ndarray

    def split(self, test_size=0.2):
        split_index = int(len(self.input_data) * (1 - test_size))
        train_data = Dataset(
            input_data=self.input_data[:split_index],
            expected_output=self.expected_output[:split_index],
        )
        test_data = Dataset(
            input_data=self.input_data[split_index:],
            expected_output=self.expected_output[split_index:],
        )
        return train_data, test_data

    def __iter__(self):
        return zip(self.input_data, self.expected_output)

    def __next__(self):
        return next(self.input_data), next(self.expected_output)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.expected_output[idx]

    def __repr__(self):
        return f"Dataset(entry[0]={next(zip(self.input_data, self.expected_output))}"

    def fuzzify(self, f=rand_dist, variance=0.1):
        """Fuzzify a dataset by up to 10%."""
        return Dataset(
            input_data=f(self.input_data, error=variance),
            expected_output=self.expected_output,
        )

    def save(self, filename, mkdirs=True):
        if mkdirs:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))

        np.savez(
            filename, input_data=self.input_data, expected_output=self.expected_output
        )

    @staticmethod
    def load(filename):
        data = np.load(filename)
        return Dataset(
            input_data=data["input_data"], expected_output=data["expected_output"]
        )
