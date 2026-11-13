d = {}
d["a"] = "s"
d["a"] = 1.5
assert d["a"] >= 1
assert d["a"] < 2
assert d["a"] <= 2
assert d["a"] != 1
assert 1 < d["a"]
assert 2 > d["a"]
assert d["a"] > 1.0
assert d["a"] == 1.5
assert not (d["a"] == 1)
assert not (d["a"] < 1.5)
assert d["a"] <= 1.5
assert d["a"] > True
d["b"] = "t"
d["b"] = 2.0
assert d["b"] == 2
assert 2 == d["b"]
assert d["b"] >= 2
assert d["b"] <= 2
assert not (d["b"] > 2)
