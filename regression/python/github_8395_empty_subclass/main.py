class A:
    pass


class B(A):
    pass


assert isinstance(B(), A)
assert not isinstance(A(), B)
