class Op:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def compute(self):
        return self.a + self.b


x = nondet_int()
y = nondet_int()
p = "ab"
q = "c"
n = Op(x, y)
s = Op(p, q)
assert n.compute() == x + y
assert s.compute() == p + q
assert len(s.compute()) == len(p) + len(q)
