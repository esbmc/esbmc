# h is k, which returns "c", when neither condition holds.
def f():
    return "a"


def g():
    return "b"


def k():
    return "c"


c1 = nondet_bool()
c2 = nondet_bool()
h = f if c1 else (g if c2 else k)
r = h()
assert r != "c"
