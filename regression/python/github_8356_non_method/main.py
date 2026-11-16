class C:
    x = 3
    y = x

assert C.y == 3
assert C().y == 3
