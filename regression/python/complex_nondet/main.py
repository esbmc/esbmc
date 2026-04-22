# Basic nondet_complex() test: verify structural properties.

z = nondet_complex()

# Constrain to finite values (exclude NaN/Inf).
__ESBMC_assume(z.real == z.real)  # NaN != NaN
__ESBMC_assume(z.imag == z.imag)

# abs(z) is always >= 0 for any finite complex number.
a = abs(z)
assert a >= 0.0
