from dataclasses import dataclass

import numpy as np


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
        return iter(zip(self.input_data, self.expected_output))

    def __next__(self):
        return next(self.input_data), next(self.expected_output)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.expected_output[idx]

    def __repr__(self):
        return f"Dataset(input_data={self.input_data}, expected_output={self.expected_output})"
