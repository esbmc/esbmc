def f4(x, y):
    u = x["string"] + y
    v = y + 3
    return u | v


w = 1
d = f4({"string": w}, 1)
assert isinstance(d, int)
assert d == 6
