from collections import defaultdict as dd

d: dict = dd(int)
d["a"] += 1
d["a"] += 1
v: int = d["missing"]
assert d["a"] == 2
