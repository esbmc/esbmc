def f(c):
    q = [3] if c else [2]
    assert isinstance(q[0], int)
    return q[0]


assert f(False) == 3
