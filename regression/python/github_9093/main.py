from collections import defaultdict
d = defaultdict(dict)
d["a"]["b"] = 1
assert "b" in d["a"]
assert "c" not in d["a"]
assert "a" in d
assert d["a"]["b"] == 1
