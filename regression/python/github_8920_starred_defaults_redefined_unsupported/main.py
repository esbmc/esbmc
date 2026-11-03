# Issue #8920: defaults are looked up by bare name, so a nested f with a
# default must not let f(*xs) bind the global f, which has none, from one
# element: CPython raises TypeError here.
def f(a, b):
    return a + b


def outer():
    def f(a, b=1):
        return a + b

    return f(1)


def t(xs: list):
    caught = False
    try:
        f(*xs)
    except TypeError:
        caught = True
    assert caught


t([1])
