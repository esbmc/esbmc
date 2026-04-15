# nondet_complex() should fail when we assert wrong constraint.
z = nondet_complex()
__ESBMC_assume(z.real == z.real)
__ESBMC_assume(z.imag == z.imag)

# This is wrong: nondet can have any finite value including negative abs.
# Actually abs is always >= 0 but this says abs == -1 which is impossible.
# The solver should find that -1 is also possible through overflow/edge behavior.
# Actually, for finite values abs(z) >= 0 is always true.
# So let's assert something clearly false: z.real == z.real + 1
assert z.real == z.real + 1.0
