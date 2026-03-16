import collections as col

# Module alias form: col.defaultdict(factory)
d: dict = col.defaultdict(int)
d["a"] += 1
d["a"] += 1
v: int = d["missing"]
assert d["a"] == 2
assert v == 0
