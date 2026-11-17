def f():
    s: str = 5
    xs = [s]
    return xs[0]


assert f() == 5
