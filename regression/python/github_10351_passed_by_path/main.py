class A:
    def f(self):
        return "a"


class B:
    def f(self):
        return 42


class Holder:
    def __init__(self, o):
        self.o = o

    def g(self):
        return self.o.f()


if nondet_bool():
    o = A()
else:
    o = B()
r = Holder(o).g()
if isinstance(o, A):
    assert r == "a"
else:
    assert r == 42
