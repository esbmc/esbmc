class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute(self):
        return self.a + self.b


s = Op("a", "b")
n = Op(1, 2)
assert isinstance(n, Op)
