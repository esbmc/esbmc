t = (0, [1.0, 1.0])
r, p = t
p.append(2.0)
assert len(p) == 3
assert p[2] == 2.0
