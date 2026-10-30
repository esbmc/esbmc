def f(x):
    return 3 * x


r = f([1, 2])
assert len(r) == 6
assert r[2] == 1
assert r[5] == 2
