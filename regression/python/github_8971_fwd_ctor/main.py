class Inner:
    def __init__(self, b):
        self.b = b


class Outer:
    def __init__(self, a, b):
        self.a = a
        self.inner = Inner(b)

    def total(self):
        return self.a + self.inner.b


flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
y = nondet_int() if flag else nondet_str()
o = Outer(x, y)
assert o.inner.b is y
assert isinstance(o.inner.b, int) == flag
