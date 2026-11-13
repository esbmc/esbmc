l = [10, 20, 30, 40, 50]
r = l[:None]
assert len(r) == 5
assert r[4] == 50
s = l[None:]
assert len(s) == 5
t = l[None:None:-1]
assert t == l[::-1]
assert t[0] == 50
u = l[None:2]
assert u == [10, 20]
v = l[1:None:None]
assert v == [20, 30, 40, 50]
