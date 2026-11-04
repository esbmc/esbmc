def h(x, y):
    u = x["string"] + y
    v = y + 3
    return u + v


w = 1
d = h({"string": w}, 1)
assert isinstance(d, int)
