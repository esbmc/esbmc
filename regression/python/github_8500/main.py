s = nondet_int()
x, y, z = (i * 2 for i in range(s, s + 3))
assert y - x == 2
