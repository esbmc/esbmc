n = nondet_int()
l = [0] * n
l.extend(l)
assert len(l) < 20
