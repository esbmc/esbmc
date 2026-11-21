d = {}
d["a"] = "s"
d["a"] = 1.5
d["b"] = "t"
d["b"] = 1
assert d["a"] < d["b"]
