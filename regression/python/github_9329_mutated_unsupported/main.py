def inner(v):
    return v


d = {"v": 1}
d["x"] = 2
try:
    inner(**d)
    assert False
except TypeError:
    pass
