class A:
    def f(self):
        return 1


class B(A):
    pass


class C(A):
    pass


b = B() if nondet_bool() else C()
assert b.f() == 1
assert not isinstance(b, A)
