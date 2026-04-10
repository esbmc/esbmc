import collections

# Module-qualified form: collections.defaultdict(factory)
d: dict = collections.defaultdict(int)
d["a"] += 1
d["a"] += 1
v: int = d["missing"]
assert d["a"] == 2
assert v == 0
