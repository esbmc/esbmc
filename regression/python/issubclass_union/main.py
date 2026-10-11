class A:
    pass


class B(A):
    pass


assert issubclass(bool, str | int)
assert not issubclass(int, str | float)
assert issubclass(int, (str, float | int))
assert issubclass(B, int | A)
assert not issubclass(A, int | B)
