w = "abcdef"
k = nondet_int()
__ESBMC_assume(k == 2 or k == 3)
r = w[::k]
assert r == "ace" or r == "ad"
