def f(c):
    q = [3] if c else [2]
    assert isinstance(q[0], int)
    return len(q)


assert f(True) == 1
assert f(False) == 1
