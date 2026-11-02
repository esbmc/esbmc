def f(c):
    q = [3] if c else [2]
    return q


a = f(True)
assert isinstance(a[0], str)
