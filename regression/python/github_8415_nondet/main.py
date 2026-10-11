n = nondet_int() % 5
assert n >= 0
assert n < 5

m = nondet_int() % -5
assert m <= 0
assert m > -5
