def inner(v):
    return v


d = {"v": 1}
assert inner(**d) == 1
