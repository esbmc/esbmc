# GitHub #8928: a nested constant index through a tuple[...] parameter
def f(t: tuple[list[float], float]):
    x = t[0][1]
    assert x == 2.0


f(([1.0, 2.0], 3.0))
