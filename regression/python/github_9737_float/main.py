def f(c):
    q = [1.5, 2.5] if c else [3.5]
    return q


a = f(True)
b = f(False)
assert isinstance(a[1], float)
assert isinstance(b[0], float)
assert len(a) == 2
