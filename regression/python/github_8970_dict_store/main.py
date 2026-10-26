# A store replaces the function a key holds.
def f():
    return "a"


def g():
    return 1


d = {"a": f, "b": f}
c = nondet_bool()
if c:
    d["a"] = g
r = d["a"]()
if c:
    assert isinstance(r, int)
    assert r == 1
else:
    assert isinstance(r, str)
    assert r == "a"
