def f():
    return 1


def g():
    return 2


def make():
    d = {"a": f, "b": g}
    return d


b = make()
r = b["b"]()
assert r == 1
