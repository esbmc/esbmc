a = [1.0, 2.0]
b = []
b.extend(a)
c = a.copy()
d = list(a)
assert b[1] == 2.0 and c[1] == 2.0 and d[1] == 1.0
