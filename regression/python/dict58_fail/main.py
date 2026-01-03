d = {"a": 1, "b": 2}
l = d.values()
keys = []
for k in l:
    keys.append(k)
assert keys == [2, 1]
