class A:
    pass


class B(A):
    pass


class C(A):
    pass


class D(B, C):
    pass


assert D.__mro__ == (D, C, B, A, object)
