assert {**{"a": 1}, **{"a": 2}}["a"] == 2

d1 = {"x": 1, "y": 2}
m = {"y": 0, **d1, "z": 6}
assert m["y"] == 2
assert len(m) == 3
keys = list(m.keys())
assert keys[0] == "y"
assert keys[1] == "x"
