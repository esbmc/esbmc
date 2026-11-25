from array import array

a = array("d", [0]) * 3
a[1] = 1.5
assert a[1] == 1.5
assert a[0] == 0.0 and a[2] == 0.0
assert len(a) == 3

r = array("i", [0]) * 3
r[1] = 2
assert r[1] == 2 and r[2] == 0
assert len(r) == 3

c = array("h", [1]) * 0
assert len(c) == 0

s = 0.0
for x in a:
    s += x
assert s == 1.5
assert a.tolist() == [0.0, 1.5, 0.0]
a.append(2)
assert a[3] == 2.0 and len(a) == 4
