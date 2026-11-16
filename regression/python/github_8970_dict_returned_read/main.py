# A def returns the call through a module-level dict.
def f():
    return "a"


def g():
    return 1


d = {"a": f, "b": g}


def pick(k):
    return d[k]()


c = nondet_bool()
if c:
    k = "a"
else:
    k = "b"
r = pick(k)
if c:
    assert isinstance(r, str)
    assert r == "a"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
