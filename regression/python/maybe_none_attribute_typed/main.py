class Op:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.total = 0

    def compute(self):
        self.total = self.a + self.b
        return self.total


op = Op(5, 10)
op.compute()
assert op.total / 3 == 5.0
