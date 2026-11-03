def k1(c):
    if c:
        return "a"
    else:
        return "b"


def k2(c):
    return "c"


def call(h, d):
    return h(d)


c = nondet_bool()
d = nondet_bool()
r = call(k1 if c else k2, d)
assert isinstance(r, str)
