# Issue #9359: an odd int above 2**53 differs from its float.
n = nondet_int()
__ESBMC_assume(n > 9007199254740992 and n < 18014398509481984)
assert n == float(n)
