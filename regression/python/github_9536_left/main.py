def f(x):
    u = x[0]
    return u * x


r = f([2, 3])
assert len(r) == 4
assert r[2] == 2
