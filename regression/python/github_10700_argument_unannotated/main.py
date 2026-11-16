def g(s):
    return s


def f():
    n = 1
    n = chr(97)
    return g(n)


assert f() == "a"
