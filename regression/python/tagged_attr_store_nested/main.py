import random


class Op:
    def __init__(self, a, b):
        self.a = a
        self.result = None
        self.inner = self.Inner(b)

    class Inner:
        def __init__(self, b):
            self.b = b

    def compute(self):
        self.result = self.a + self.inner.b
        return self.result


if random.random() < 0.5:
    x, y = 1, 2
else:
    x, y = "s", "t"
op = Op(x, y)
assert op.result is None
assert op.compute() == x + y
assert op.result == x + y
