d: dict[str, dict[str, int]] = {}
d["a"] = {}
d["a"]["b"] = 1
d["a"] = {"c": 2}
assert "b" in d["a"]
