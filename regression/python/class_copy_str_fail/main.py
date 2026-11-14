class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute(self):
        return self.a + self.b


n = Op(1, 2)
s = Op("ab", "c")
assert n.compute() == 3
assert s.compute() == "cab"
