def inner(v):
    return v


d = {"v": 1, "x": 2}
assert inner(**d) == 1
