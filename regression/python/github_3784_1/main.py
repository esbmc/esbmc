# Test dict.pop(key) - basic: add then remove a key
d: dict[str, int] = {"a": 1}
d["b"] = 2
v = d.pop("b")
assert v == 2
assert "b" not in d
assert "a" in d
assert d["a"] == 1
