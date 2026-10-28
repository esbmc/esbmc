# github #9331: a call forwarding *args and **kw
def inner(v):
    return v


def outer(*args, **kw):
    return inner(*args, **kw)


def outer_kw(**kw):
    return inner(**kw)


assert outer(1) == 1
assert outer(v=2) == 2
assert outer_kw(v="a") == "a"
