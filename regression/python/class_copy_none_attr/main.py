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
assert n.result is None and s.result is None
r = n.compute()
q = s.compute()
assert isinstance(r, int)
assert isinstance(q, str)
assert r == 3
assert q == "ab"
assert n.result == 3
