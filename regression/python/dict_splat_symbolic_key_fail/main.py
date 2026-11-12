d1 = {"a": 1, "b": 2}
k = input()
d2 = {k: 3}
merged = {**d1, **d2}
assert merged[k] == 4
