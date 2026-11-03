class A:
    pass


assert not isinstance(int, str)
assert not isinstance(bool, str)
assert not isinstance(A, str)
assert not isinstance(int, int)
assert not isinstance(A, A)
assert not isinstance(str, (int, str))
assert isinstance(str, object)
assert isinstance(A, (str, type))
assert isinstance("int", str)
