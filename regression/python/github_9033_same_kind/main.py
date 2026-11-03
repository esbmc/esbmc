def f():
    return "a"


def g():
    return "b"


c = nondet_bool()
r = f() if c else g()
assert isinstance(r, str)
assert (r == "a") == c
