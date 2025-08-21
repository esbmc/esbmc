a = [1] * 5
assert a[0] == 1
assert a[1] == 1
assert a[2] == 1
assert a[3] == 1
assert a[4] == 1

b = [1.5] * 4
assert b[0] == 1.5
assert b[1] == 1.5
assert b[2] == 1.5
assert b[3] == 1.5

c = [float("inf")] * 5
assert c[0] == float("inf")
assert c[1] == float("inf")
assert c[2] == float("inf")
assert c[3] == float("inf")

n = 3
d = [2] * 3
assert d[0] == 2
assert d[1] == 2
assert d[2] == 2