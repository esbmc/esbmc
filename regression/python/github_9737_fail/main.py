def f(c):
    q = [3] if c else [2]
    assert isinstance(q[0], str)
    return len(q)


assert f(True) == 1
