class A:
    pass


assert isinstance(int, type)
assert isinstance(A, type)
x = int
assert isinstance(x, (type, str))
assert isinstance(x, object)
