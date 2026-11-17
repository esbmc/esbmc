class A:
    def f(self):
        return 42


class B(A):
    pass


class C(A):
    def f(self):
        return "c"


inherit = nondet_bool()
o = B() if inherit else C()
m = o.f
r = m()
assert isinstance(o, A)
if inherit:
    assert r + 1 == 43
else:
    assert r == "c"
