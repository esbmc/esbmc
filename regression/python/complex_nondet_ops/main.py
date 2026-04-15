# nondet_complex() with component access and conjugate identity.

z = nondet_complex()

# Constrain to finite values.
__ESBMC_assume(z.real == z.real)
__ESBMC_assume(z.imag == z.imag)

# Access real and imaginary parts.
r = z.real
i = z.imag

# Conjugate identity: z.conjugate().conjugate() components match z's.
c = z.conjugate()
cc = c.conjugate()
assert cc.real == z.real
assert cc.imag == z.imag
