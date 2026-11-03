class B:
    def __init__(self):
        self.child = self.func2

    def func(self):
        return self.child()

    def func2(self):
        return 42

assert B().func() == 42
