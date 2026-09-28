# A heterogeneous dict's value type must be read per-key, not from a single
# stale answer: d["b"] is a float (2.5), not the int recorded for d["a"].
d = {"a": 1, "b": 2.5}
assert d["b"] == 1
