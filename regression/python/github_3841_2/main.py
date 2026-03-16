from collections import defaultdict

# Frequency counting pattern using AugAssign (d[key] += 1)
d: dict = defaultdict(int)
d["a"] += 1
d["a"] += 1
d["b"] += 1
assert d["a"] == 2
assert d["b"] == 1
