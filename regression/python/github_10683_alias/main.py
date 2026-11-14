# h is get, so h()() calls g, as use_g holds.
use_g = True


def g():
    return 1


def get():
    if use_g:
        a = g
    else:
        a = 7
    return a


h = get
r = h()()
assert r == 1
