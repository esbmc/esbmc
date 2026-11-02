def f(c):
    q = [3] if c else [2]
    return q


b = f(False)
assert b == [3]
