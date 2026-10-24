# len(range(n)) is the number of elements the range yields. For a negative n
# that is 0, not n, so the original form of this test asserted something
# CPython does not hold: it passed only while the symbolic single-argument
# range set its size to n without clamping. The bound is constrained here so
# the intended claim, len == n, is the one being checked.
n = nondet_int()
__ESBMC_assume(n >= 0)
x = len(range(n))
assert x == n
