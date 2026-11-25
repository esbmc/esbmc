d = {}
d["a"] = "s"
d["a"] = 2
assert d["a"] > 1.5
assert d["a"] < 2.5
assert d["a"] == 2.0
assert d["a"] != 2.5
assert 1.5 < d["a"]
