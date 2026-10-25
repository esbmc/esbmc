# GitHub #8928: a constant index into a tuple[...] parameter has the member type
def f(t: tuple[list[float], float]):
    r = t[0]
    m = t[1]
    assert r[1] * m == 3.0


f(([1.0, 2.0], 3.0))
