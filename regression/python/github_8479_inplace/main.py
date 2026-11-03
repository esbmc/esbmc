a = {"x": 1}
b = {"x": 2, "y": 3}
c = a
a |= b
assert c["x"] == 2
assert c["y"] == 3
assert len(c) == 2
