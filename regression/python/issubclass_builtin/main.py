class A:
    pass
class B(A):
    pass
assert issubclass(bool, int)
assert issubclass(B, A)
assert not issubclass(A, B)
