import lib

x = nondet_int()
__ESBMC_assume(x > -1000)
b = lib.f(x)
assert b >= 0 or b < 0
assert lib.f(-3) == 3
assert lib.neg(2) == -2
