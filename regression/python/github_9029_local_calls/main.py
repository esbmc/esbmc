def f():
    return "a"


def g():
    return 1


def call(h):
    if h == 0:
        r = f()
    else:
        r = g()
    return r


assert call(0) == "a"
assert call(1) == 1
