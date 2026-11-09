d: dict[str, str] = {"a": "xa", "b": "xb"}
r = d["a"] + d["b"]
assert r == "xaxb"
assert d["a"] != d["b"]
assert d["a"] == d["a"]

e = {"a": "x", "b": "y"}
assert e["a"] + e["b"] == "xy"
