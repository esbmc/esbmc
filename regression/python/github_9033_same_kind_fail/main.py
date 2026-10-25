def f():
    return "a"


def g():
    return "b"


c = nondet_bool()
r = f() if c else g()
assert r == "a"
