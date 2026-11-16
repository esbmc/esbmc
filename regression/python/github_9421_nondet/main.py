a = nondet_int()
__ESBMC_assume(a >= 0 and a <= 3)
assert len(range(a, 4)) == 4 - a
assert len(range(a, 10, 3)) == (10 - a + 2) // 3
