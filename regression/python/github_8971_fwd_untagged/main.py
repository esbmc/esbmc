class Inner:
    def __init__(self, b):
        self.b = b


class Outer:
    def __init__(self, a, b):
        self.a = a
        self.inner = Inner(b)


n = nondet_int()
o = Outer(n, 2)
assert o.inner.b == 2
assert o.a == n
