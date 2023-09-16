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
    def train(
        model: Sequential, dataset: Dataset, config: TrainingConfig = TrainingConfig()
    ):
        for epoch in range(config.epochs):
            output = model.forward(dataset.input_data)
            mse = model.backward(dataset.expected_output, config.learning_rate)

            if epoch % (int(config.epochs * 0.1) or 1000) == 0:
                print(f"Epoch {epoch}, MSE: {mse}")

        print("Final Weights:")
        for i, w in enumerate(model.weights):
            print(f"Layer {i+1} weights:")
            print(w)

        print("Final and Expected Output:")
        for o, e in zip(output, dataset.expected_output):
            print(f"Final: {o}, Expected: {e}")
            model.set_training_error(mse)
