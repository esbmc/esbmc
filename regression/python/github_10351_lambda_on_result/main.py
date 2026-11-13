class A:
    def f(self):
        return "a"


class B:
    def f(self):
        return 42


g = lambda v: v
if nondet_bool():
    o = A()
else:
    o = B()
r = g(o.f())
if isinstance(o, A):
    assert r == "a"
else:
    assert r == 42
