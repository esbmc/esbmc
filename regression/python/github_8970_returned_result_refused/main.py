# The result of h() is stored before it is returned.
def f():
    return "a"


def g():
    return 1


def call(h):
    r = h()
    return r


c = nondet_bool()
r = call(f if c else g)
assert (r == "a") == c
