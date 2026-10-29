# The result of pick is called in the same statement.
def f(x):
    return "a"


def g(x):
    return 1


def pick(c):
    if c:
        return f
    return g


c = nondet_bool()
r = pick(c)(0)
if c:
    assert type(r) == str
    assert r.upper() == "A"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
