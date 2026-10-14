# Negative variant: the div/mod identity q*b + r == a holds exactly, so
# asserting it is off by one must FAIL (pins the IREP2 sign-correction trees).
a: int = __VERIFIER_nondet_int()
b: int = __VERIFIER_nondet_int()
__ESBMC_assume(a >= -10 and a <= 10)
__ESBMC_assume(b >= 1 and b <= 5)
q = a // b
r = a % b
assert q * b + r == a + 1
