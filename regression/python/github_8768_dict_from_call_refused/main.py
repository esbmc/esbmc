from esbmc import nondet_bool


def f():
    return "a"


def g():
    return 1


def make():
    return {"a": f, "b": g}


d = make()
k = "a" if nondet_bool() else "b"
r = d[k]()
if k == "b":
    assert r == 1
