class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.result = None

    def compute(self):
        self.result = self.a + self.b
        return self.result


x = nondet_int()
y = nondet_int()
n = Op(x, y)
s = Op("a", "b")
r = n.compute()
q = s.compute()
assert isinstance(r, int)
assert isinstance(q, str)
assert r == x + y
assert n.result == r
