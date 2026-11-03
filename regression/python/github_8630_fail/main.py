def f(a):
    return a + 1


def g(a):
    return a + 1


xs = [1, 2, 3]
c = [f(a) for a in [g(b) for b in xs]]
assert isinstance(c[0], float)
