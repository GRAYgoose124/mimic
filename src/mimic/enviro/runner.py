import logging
from typing import Callable

log = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, M, DS, tests=None):
        self.M = M
        self.DS = DS

        self._tests = []
        if tests is not None:
            for test in tests:
                if isinstance(test, Callable):
                    self.add_test(test)
                else:
                    log.warning(f"Test {test} is not callable and will be ignored.")

    @property
    def tests(self):
        return self._tests

    def add_test(self, test):
        self._tests.append(test)

    def test(self):
        try:
            for test in self.tests:
                test(self.M, self.DS)

            log.info(
                f"Passed! {self.M.error_fn.__class__.__name__}: {self.M.training_error}"
            )
            return True
        except AssertionError as e:
            print(e)
            return False
