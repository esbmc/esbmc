def g(s):
    return s


def f():
    n: int = chr(97)
    return g(n)


assert f() == "a"
