from dataclasses import dataclass


from .dataset import Dataset
from .models.sequential import Sequential


@dataclass
class TrainingConfig:
    learning_rate: float = 0.5
    epochs: int = 10000
    momentum: float = 0.0


class Trainer:
    @staticmethod
    def train(M: Sequential, DS: Dataset, C: TrainingConfig = TrainingConfig()):
        for epoch in range(C.epochs):
            output = M.forward(DS.input_data)
            error = M.backward(DS.expected_output, C.learning_rate)

            if epoch % (int(C.epochs * 0.1) or 1000) == 0:
                print(f"Epoch {epoch}, {M.error_fn.__class__.__name__}: {error}")

        print("Final Weights:")
        for i, w in enumerate(M.weights):
            print(f"Layer {i+1} weights:")
            print(w)

        print("Final and Expected Output:")
        for o, e in zip(output, DS.expected_output):
            print(f"Final: {o}, Expected: {e}")
            M.set_training_error(error)
