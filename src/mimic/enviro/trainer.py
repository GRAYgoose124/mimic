from dataclasses import dataclass

import numpy as np


from ..utils.dataset import Dataset
from ..models.sequential import Sequential
from ..utils.weights import total_norm


@dataclass
class TrainingConfig:
    learning_rate: float = 0.5
    epochs: int = 10000
    momentum: float = 0.0
    post_normalize: bool = False
    normalize: bool = False


class Trainer:
    @staticmethod
    def train(M: Sequential, DS: Dataset, C: TrainingConfig = TrainingConfig()):
        for epoch in range(C.epochs):
            # forward and backward pass
            output = M.forward(DS.input_data)
            error = M.backward(DS.expected_output, C.learning_rate)

            if C.normalize:
                total_norm(M.weights)

            if epoch % (int(C.epochs * 0.1) or 1000) == 0:
                print(f"Epoch {epoch}, {M.error_fn.__class__.__name__}: {error}")
        M.set_training_error(error)

        if C.post_normalize:
            # normalize all weights in all layers
            total_norm(M.weights)

        print("Final Weights:")
        for i, w in enumerate(M.weights):
            print(f"Layer {i+1} weights:\n{w}")

        print("Final and Expected Output:")
        for o, e in zip(output, DS.expected_output):
            print(f"Final: {o}, Expected: {e}")
