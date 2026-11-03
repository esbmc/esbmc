m = {"x": 1}
b = {"y": 2, "x": 3}
m.update(b)
assert m["x"] == 3
assert m["y"] == 2
assert len(m) == 2
