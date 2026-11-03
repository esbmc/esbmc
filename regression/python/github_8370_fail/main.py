def g():
    return 42

class B:
    def __init__(self):
        self.child = g

    def func(self):
        return self.child()

assert B().func() == 41
