def k1(c):
    if c:
        return "a"
    else:
        return 1


def k3(c):
    return "p"


def call(h, d):
    return h(d)


c = nondet_bool()
d = nondet_bool()
r = call(k1 if c else k3, d)
assert isinstance(r, str) == (c == d or not c)
