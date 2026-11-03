def f(c):
    q = [1.5, 2.5] if c else [3.5]
    assert isinstance(q[0], float)
    return len(q)


assert f(True) == 2
assert f(False) == 1
