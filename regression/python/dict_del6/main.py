d = {"x": 10}
del d["x"]
d["x"] = 20
assert "x" in d         # Should be True
