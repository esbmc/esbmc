from collections import defaultdict
d = defaultdict(dict)
d["a"]["b"] = 1
assert "b" not in d["a"]
