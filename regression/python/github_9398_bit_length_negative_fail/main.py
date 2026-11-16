# Issue #9398: a negative int has a nonzero bit_length.
x: int = nondet_int()
__ESBMC_assume(x < 0)
assert x.bit_length() == 0
