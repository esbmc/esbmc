a = [x for x in range(4)]
assert len(a) == 4
assert a[3] == 3

b = [x * 2 for x in [1, 2, 3]]
assert b[2] == 6

c = [x for x in range(6) if x % 2 == 0]
assert len(c) == 3
assert c[2] == 4

d = [x for x in range(6) if x > 1 if x < 4]
assert len(d) == 2
assert d[0] == 2

e = [p + q for p, q in [(1, 2), (3, 4)]]
assert e[1] == 7

f = [[y for y in range(2)] for x in range(2)]
assert f[1][1] == 1
