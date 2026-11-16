# The key picks the function, so each path sees the type of its own result.
def f():
    return "a"


def g():
    return 1


d = {"a": f, "b": g}
c = nondet_bool()
k = "a" if c else "b"
r = d[k]()
if c:
    assert isinstance(r, str)
    assert r == "a"
else:
    assert isinstance(r, int)
    assert r + 1 == 2
assert d["a"]() == "a"
