# nondet_complex() with assume constraints.

z = nondet_complex()
__ESBMC_assume(z.real >= 0.0)
__ESBMC_assume(z.real <= 1.0)
__ESBMC_assume(z.imag >= 0.0)
__ESBMC_assume(z.imag <= 1.0)

# With these constraints, z + z should have components in [0, 2].
w = z + z
assert w.real >= 0.0
assert w.real <= 2.0
assert w.imag >= 0.0
assert w.imag <= 2.0
