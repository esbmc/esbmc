t = (10, 20, 30, 40, 50)
s = 2
r = t[::s]
assert len(r) == 3
assert r[1] == 30
