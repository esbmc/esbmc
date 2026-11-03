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
    return "p"


d = nondet_bool()
assert isinstance(call(plain, d), str)
r = call(k1, d)
assert isinstance(r, str) == d
