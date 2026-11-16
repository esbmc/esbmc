# a is the function that pick returns, so it is f or g, which return str and int.
def f():
    return "a"


def g():
    return 1


def pick(c):
    if c:
        return f
    return g


c = nondet_bool()
a = pick(c)
r = a()
if c:
    assert type(r) == str
    assert r.upper() == "A"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
assert (a is f) == c
