# setdefault and update add functions, and setdefault keeps a function that is there.
def f():
    return "a"


def g():
    return 1


d = {"a": f}
d.setdefault("a", g)
d.setdefault("b", g)
d.update({"c": f})
c = nondet_bool()
k = "b" if c else "c"
r = d[k]()
if c:
    assert isinstance(r, int)
    assert r == 1
else:
    assert isinstance(r, str)
    assert r == "a"
assert d["a"]() == "a"
