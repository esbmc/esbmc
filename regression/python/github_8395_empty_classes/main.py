class A:
    pass


class B:
    pass


assert not isinstance(A(), B)
assert not isinstance(B(), A)
assert isinstance(A(), A)
assert isinstance(B(), B)
assert not isinstance(A(), int)
