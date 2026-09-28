a = nondet_bytes(4)
b = nondet_bytes(3)
c = a + b
assert len(c) == 7
assert c[0] == a[0]
assert c[4] == b[0]
assert c[6] == b[2]
