def inner(v, **kw):
    return v + len(kw)


d = {"v": 1, "x": 2}
assert inner(**d) == 2
