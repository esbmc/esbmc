d1 = {"a": 1, "b": 2}
k = input()
d2 = {k: 3}
merged = {**d1, **d2}
assert k in merged
assert merged[k] == 3
assert len(merged) == (2 if k in d1 else 3)
