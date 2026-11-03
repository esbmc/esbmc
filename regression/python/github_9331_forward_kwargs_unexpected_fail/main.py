# github #9331: a keyword forwarded through **kw that inner() does not take
def inner(v):
    return v


def outer(*args, **kw):
    return inner(*args, **kw)


assert outer(1) == 1
outer(1, w=3)
