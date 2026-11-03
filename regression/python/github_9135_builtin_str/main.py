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
r = call(k1 if c else str, d)
assert isinstance(r, str) == ((not c) or d)
