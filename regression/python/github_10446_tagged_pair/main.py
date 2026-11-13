d = {}
d["a"] = "s"
d["a"] = 1.5
d["b"] = "t"
d["b"] = 1
d["c"] = "u"
d["c"] = 1.0
assert d["a"] > d["b"]
assert d["b"] < d["a"]
assert d["c"] == d["b"]
assert d["a"] != d["b"]
assert d["c"] >= d["b"]
