class A:
    def f(self):
        return 1


class B:
    pass


if nondet_bool():
    o = A()
elif nondet_bool():
    o = B()
else:
    o = A()
assert o.f() == 1
