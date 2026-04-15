# nondet_complex() with abs — verify abs is always non-negative
# when finite values are assumed.
z = nondet_complex()
__ESBMC_assume(z.real == z.real)
__ESBMC_assume(z.imag == z.imag)

a = abs(z)
assert a >= 0.0

# Conjugate of nondet: z.conjugate().imag == -z.imag
c = z.conjugate()
assert c.real == z.real
