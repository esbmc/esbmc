def inner(v):
    return v


for i in range(3):
    d = {"v": i}
    assert inner(**d) == i
