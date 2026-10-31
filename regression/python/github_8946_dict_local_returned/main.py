def f():
    return "a"


def g():
    return 1


def make():
    d = {"a": f, "b": g}
    return d


b = make()
r = b["b"]()
assert r == 1
