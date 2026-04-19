a = nondet_str()
b = nondet_str()
c = nondet_str()
d = {0: a, 1: b, 2: c}
assert d[0] == a
assert d[1] == b
assert d[2] == c
