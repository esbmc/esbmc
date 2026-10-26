from esbmc import nondet_bool


def f1():
    return 42


def f2():
    return "a"


d = {"a": {"b": f1}}
r = nondet_bool()
if r:
    d["a"] = {"b": f2}
e = d["a"]["b"]()
if r:
    assert e == "a"
else:
    assert e == 42
