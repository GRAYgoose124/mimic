
from mimic.layers.dense import Dense
from mimic.models.sequential import Sequential
from mimic.utils.data import xor_set


def test_sequential_dense():
    model = Sequential([Dense(2),
                    Dense(4),
                    Dense(4),
                    Dense(1)])

    model.train(xor_set, epochs=1000, learning_rate=0.01, momentum=0.01)