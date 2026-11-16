c = False
q = [1.5, 2.5] if c else [3.5]
assert isinstance(q[0], float)
assert len(q) == 1
r = [3] + ([4] if c else [5])
assert isinstance(r[1], int)
assert r == [3, 5]
