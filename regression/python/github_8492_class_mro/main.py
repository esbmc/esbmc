class A:
    pass


class B(A):
    pass


class C(A):
    pass


class D(B, C):
    pass


assert D.__mro__ == (D, B, C, A, object)
assert B.__mro__ == (B, A, object)
m = A.__mro__
assert m == (A, object)
assert len(D.__mro__) == 5
