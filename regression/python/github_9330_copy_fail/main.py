class A:
    pass


class B(A):
    pass


class C:
    pass


o = B() if nondet_bool() else C()
p = o
assert isinstance(p, A)
