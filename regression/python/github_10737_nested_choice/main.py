# h is one of three functions of one type, chosen by a nested conditional.
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
assert r == ("a" if c1 else ("b" if c2 else "c"))
