class A:
    pass


class B(A):
    pass


assert issubclass(bool, str | int)
assert not issubclass(float, int | str)
assert issubclass(B, int | A)
assert not issubclass(A, int | None | B)
