d = {"a": 1, "b": 2}
values = 0
for k in d:
    values = values + d[k]
assert values == 4
