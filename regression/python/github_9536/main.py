def f(x, n):
    return x * n


assert len(f([2, 3], 2)) == 4
r = f([2, 3], 2)
assert r[3] == 3
assert f([2, 3], 2) == [2, 3, 2, 3]
