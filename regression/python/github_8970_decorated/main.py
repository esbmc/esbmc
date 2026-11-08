# The decorated `a` returns an int, `b` a str.
def plus_one(f):
    def inner(x):
        return x + 1

    return inner


@plus_one
def a(x):
    return "a"


def b(x):
    return "b"

c = nondet_bool()
h = a if c else b
r = h(1)
if c:
    assert r == 2
else:
    assert r == "b"
