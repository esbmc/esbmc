# Symbolic Python floor-division and modulo with mixed signs exercise the
# sign-correction trees in python_math.cpp whose `value < 0` sign tests are
# built in IREP2 with gen_typecast_arithmetic width reconciliation. The div/mod
# identity and remainder range must hold for a negative dividend.
a: int = __VERIFIER_nondet_int()
b: int = __VERIFIER_nondet_int()
__ESBMC_assume(a >= -10 and a <= 10)
__ESBMC_assume(b >= 1 and b <= 5)
q = a // b
r = a % b
assert q * b + r == a
assert r >= 0 and r < b
