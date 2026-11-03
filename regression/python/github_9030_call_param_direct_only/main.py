def mixed(c):
    if c:
        return "a"
    else:
        return 1


def plain(c):
    return "p"


def call(h, d):
    return h(d)


d = nondet_bool()
assert isinstance(mixed(d), str) == d
assert isinstance(call(plain, d), str)
