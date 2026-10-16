a = {"x": 1}
b = {"y": 3}
c = a
a |= b
assert len(c) == 1
