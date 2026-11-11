import array

a = array.array("d", [0.0, 1.5])
assert a[1] == 1.5
c = a * 2
assert len(c) == 4 and c[3] == 1.5
