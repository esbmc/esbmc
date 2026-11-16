def g() -> int:
    return 3


x = 2
z = 0
d = {}
d["a"] = g()
d["b"] = x + 1
d["c"] = z
assert isinstance(d["a"], int)
assert isinstance(d["b"], int)
assert d["a"] == d["b"]
assert not d["c"]
