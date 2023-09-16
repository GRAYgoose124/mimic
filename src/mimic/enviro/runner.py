class ModelRunner:
    def __init__(self, M, DS):
        self.M = M
        self.DS = DS

        self._tests = []

    @property
    def tests(self):
        return self._tests

    def add_test(self, test):
        self._tests.append(test)

    def test(self):
        try:
            for test in self.tests:
                test(self.M, self.DS)

            print(
                f"Passed! {self.M.error_fn.__class__.__name__}: {self.M.training_error}"
            )
            return True
        except AssertionError as e:
            print(e)
            return False
