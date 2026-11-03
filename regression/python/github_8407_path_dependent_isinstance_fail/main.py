class A:
    pass


class B(A):
    pass


class C(A):
    pass


b = B() if nondet_bool() else C()
assert isinstance(b, B)
