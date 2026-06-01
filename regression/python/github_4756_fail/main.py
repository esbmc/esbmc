x = nondet_int()
__ESBMC_assume(1 <= x)
__ESBMC_assume(x <= 16)
b = x.bit_length()
assert b == 99  # must fail: bit_length(x in [1,16]) is in [1,5]
