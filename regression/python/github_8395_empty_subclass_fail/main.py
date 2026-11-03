class A:
    pass


class B(A):
    pass


assert not isinstance(B(), A)
