# nondet_complex() arithmetic: z - z should always be zero for finite z.
z = nondet_complex()
__ESBMC_assume(z.real == z.real)
__ESBMC_assume(z.imag == z.imag)
# Also exclude infinity (inf - inf = NaN).
__ESBMC_assume(z.real < 1e300)
__ESBMC_assume(z.real > -1e300)
__ESBMC_assume(z.imag < 1e300)
__ESBMC_assume(z.imag > -1e300)

w = z - z
assert w.real == 0.0
assert w.imag == 0.0
