def h(n):
    return n * 2


def g(v):
    return v + 1


def func(a, n):
    return a(h(n))


c = func
d = c(g, 3)
assert d == 7
