import lib

b = lib.f(nondet_int())
assert b >= 0 or b < 0
assert lib.f(-3) == -3
