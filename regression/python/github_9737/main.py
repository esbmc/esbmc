def f(c):
    q = [3] if c else [2]
    return q


a = f(True)
b = f(False)
assert isinstance(a[0], int)
assert isinstance(b[0], int)
assert a == [3]
assert b == [2]
