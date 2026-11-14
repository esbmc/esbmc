k = input()
d1 = {"a": 1}
m = {"b": 0, **d1, k: 5}
assert m[k] == 5
assert m["a"] == (5 if k == "a" else 1)
assert len(m) == (3 if k != "a" and k != "b" else 2)
