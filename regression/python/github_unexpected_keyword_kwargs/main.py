def inner(v, **kw):
    return v + len(kw)


assert inner(v=1, x=2) == 2
