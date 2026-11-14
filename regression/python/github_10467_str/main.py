# Issue #10467: a non-literal str slice step was sliced with step 1.
s: int = nondet_int()
__ESBMC_assume(s == 2)
t = "abcde"
assert t[::s] == "abcde"
