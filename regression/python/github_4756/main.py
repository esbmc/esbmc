x = nondet_int()
__ESBMC_assume(1 <= x)
__ESBMC_assume(x <= 16)
b = x.bit_length()
assert b >= 1
assert b <= 5
