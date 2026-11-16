def dec(f):

    def w(x):
        return f(x) + 1

    return w


@dec
def h(x):
    return x


def half(x):
    return x / 2


q = dec(half)

assert h(1) == 2
assert q(1) == 1.5
