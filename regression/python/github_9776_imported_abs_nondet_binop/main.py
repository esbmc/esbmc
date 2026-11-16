import lib

b = lib.f(nondet_int() % 1000 - 500)
assert b >= 0
