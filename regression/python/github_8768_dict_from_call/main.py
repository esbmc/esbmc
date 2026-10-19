from esbmc import nondet_bool


def f():
    return 1


def g():
    return 2


def make():
    return {"a": f, "b": g}


d = make()
k = "a" if nondet_bool() else "b"
r = d[k]()
if k == "b":
    assert r == 2
else:
    assert r == 1
