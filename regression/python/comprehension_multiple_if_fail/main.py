# Both conditions must hold: 0 and 1 are excluded by `if x > 1`.
r = [x for x in range(6) if x > 1 if x < 5]
assert r == [0, 1, 2, 3, 4]
