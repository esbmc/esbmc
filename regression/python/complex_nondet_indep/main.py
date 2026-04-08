# nondet_complex() components are independent: constraining one
# does not affect the other.
z = nondet_complex()
__ESBMC_assume(z.real == z.real)  # exclude NaN
__ESBMC_assume(z.imag == z.imag)
__ESBMC_assume(z.real == 5.0)
# z.imag is still unconstrained (but finite)
assert z.real == 5.0
