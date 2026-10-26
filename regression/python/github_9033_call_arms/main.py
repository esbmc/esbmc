def f():
    return "a"


def g():
    return 1


c = nondet_bool()
r = f() if c else g()
assert isinstance(r, str) == c
assert isinstance(r, int) != c
