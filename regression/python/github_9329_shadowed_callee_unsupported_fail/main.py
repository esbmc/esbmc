def inner(v):
    return v


try:
    raise ValueError
except ValueError as inner:
    pass
assert inner(**{"v": 1}) == 1
