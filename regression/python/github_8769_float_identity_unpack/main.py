from esbmc import nondet_int, nondet_float, nondet_str

a = [nondet_int(), nondet_float(), nondet_str()]
b, c, d = a
assert b is a[0]
assert c is a[1]
assert d is a[2]
