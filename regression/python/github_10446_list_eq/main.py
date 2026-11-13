d = {}
d["a"] = "s"
d["a"] = 1.0
e = {}
e["a"] = "t"
e["a"] = 1
assert [d["a"]] == [e["a"]]
