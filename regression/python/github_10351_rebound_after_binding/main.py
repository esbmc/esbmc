class A:
    def f(self):
        return "a"


class B:
    def f(self):
        return 42


o = A()
if nondet_bool():
    o = B()
r = o.f()
if isinstance(o, A):
    assert r == "a"
else:
    assert r == 42
