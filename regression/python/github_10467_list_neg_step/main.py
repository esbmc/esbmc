l = [10, 20, 30, 40, 50]
s = -1
r = l[::s]
assert len(r) == 5
assert r[0] == 50
assert r[4] == 10
t = -2
q = l[::t]
assert len(q) == 3
assert q[0] == 50
assert q[1] == 30
assert q[2] == 10
