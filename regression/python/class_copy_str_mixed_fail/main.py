class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute(self):
        return self.a + self.b


n = Op(1, 2)
s = Op("a", 2)
assert n.compute() == 3
r = s.compute()
