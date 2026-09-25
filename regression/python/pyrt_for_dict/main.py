d = {"a": 1, "b": 2}
keys = 0
values = 0
for k in d:
    keys = keys + 1
    values = values + d[k]
assert keys == 2
assert values == 3
