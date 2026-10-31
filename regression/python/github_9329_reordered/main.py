def inner(v, w=3):
    return v + w


d = {"w": 4, "v": 1}
assert inner(**d) == 5
