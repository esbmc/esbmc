l = [1.5, 2.5]
r = l[0:1]
r.append(3)
c = list(l)
c.append(4)
assert r[0] == 1.5
assert len(r) == 2
assert c[1] == 2.5
