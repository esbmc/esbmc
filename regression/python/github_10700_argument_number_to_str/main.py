def g(s):
    return s


def f():
    n: str = 5
    return g(n)


assert f() == 5
