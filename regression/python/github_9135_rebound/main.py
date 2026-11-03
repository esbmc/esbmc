def k1(c):
    if c:
        return "a"
    else:
        return 1


def k2(c):
    if c:
        return 2
    else:
        return "b"


def call(h, d):
    return h(d)

c = nondet_bool()
d = nondet_bool()
f = k1 if c else k2
from helper import plain as f
r = call(f, d)
assert not isinstance(r, str)
