# github #9331: a key given both by name and through **kw
def inner(v=0, w=0):
    return v + w


def outer(**kw):
    return inner(v=1, **kw)


assert outer(w=2) == 3
try:
    outer(v=2)
    raised = False
except TypeError:
    raised = True
assert raised
