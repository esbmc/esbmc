import lib

x = nondet_int()
__ESBMC_assume(x > -1000 and x < 1000)
b = lib.f(x)
assert b >= 0
assert b == x or b == -x
