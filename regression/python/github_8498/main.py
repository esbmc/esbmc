def s(a, *rest):
    return a + len(rest)


def call(f):
    return f(1, 2, 3)


def call_one(g, x):
    return g(x)


assert call(s) == 3
assert call_one(s, 4) == 4
assert s(5, 6) == 6
