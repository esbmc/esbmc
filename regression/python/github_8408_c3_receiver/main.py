class A:

    def func(self):
        return "a"


class B(A):
    pass


class C(A):

    def func(self):
        return 42


class D(B, C):
    pass


c = nondet_bool()
if c:
    d = D()
else:
    d = B()
e = d.func()
if c:
    assert e == 42
else:
    assert e == "a"
