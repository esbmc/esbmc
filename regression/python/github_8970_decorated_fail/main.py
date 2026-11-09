# On the `b` path `r` is "b", so the assertion fails.
def plus_one(f):
    def inner(x):
        return x + 1

    return inner


@plus_one
def a(x):
    return "a"


def b(x):
    return "b"

h = a if nondet_bool() else b
r = h(1)
assert r == 2
