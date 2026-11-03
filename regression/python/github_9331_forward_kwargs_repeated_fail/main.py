# github #9331: a key given both by name and through **kw
def inner(v=0, w=0):
    return v + w


def outer(**kw):
    return inner(v=1, **kw)


assert outer(w=2) == 3
assert outer(v=2) == 3
