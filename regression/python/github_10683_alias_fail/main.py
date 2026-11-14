# h()() calls 7 when use_g does not hold, which raises TypeError.
use_g = nondet_bool()


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
