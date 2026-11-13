a = [1.0, 2.0]
b = []
b.extend(a)
assert b[1] == 2.0
c = a.copy()
assert c[1] == 2.0
d = list(a)
assert d[1] == 2.0
