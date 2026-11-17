class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.result = None

    def compute(self):
        self.result = self.a + self.b
        return self.result


n = Op(1, 2)
s = Op("a", "b")
r = n.compute()
q = s.compute()
assert isinstance(r, int)
assert r == 4
