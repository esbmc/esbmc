def g() -> int:
    return 3


z = 0
d = {}
d["a"] = g()
d["c"] = z
assert d["c"]
