# Test dict.pop(key, default) - key present and absent
d: dict[str, int] = {"x": 10, "y": 20}
v1 = d.pop("x", 99)
assert v1 == 10
assert "x" not in d

v2 = d.pop("z", 42)
assert v2 == 42
assert "z" not in d
assert "y" in d
assert d["y"] == 20
