d = {"a": 1, "b": 2}
l = d.keys()
keys = []
for k in l:
    keys.append(k)
assert keys == ["b", "a"]
