xs: list[int] = []
n = 0
r = n == 0 and (not xs or xs[0] > 0)
assert r
s = n == 1 or (len(xs) > 0 and xs[0] > 0)
assert not s
assert n == 0 and (not xs or xs[0] > 0)
