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
assert isinstance(o, B)
