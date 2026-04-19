d = {"a": 1}
del d["a"]
d["b"] = 2
assert "b" in d  # Should be True
