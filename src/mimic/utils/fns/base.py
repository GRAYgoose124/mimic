from abc import ABC, abstractmethod

import numpy as np

# TODO: np.vectorize fn/derivative


class DifferentFn(ABC):
    @abstractmethod
    def fn(self):
        pass

    @abstractmethod
    def derivative(self):
        pass
