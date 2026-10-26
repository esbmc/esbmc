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

def plain(x):
    return 3


def apply(cb, g, d):
    return cb(d) + g(d)


c = nondet_bool()
d = nondet_bool()
r = call(k1 if c else k2, d)
assert isinstance(r, str) == (c == d)
assert apply(plain, plain, d) == 7
