# Issue #10494: len(x) - k was computed unsigned and wrapped.
l = nondet_list()
__ESBMC_assume(len(l) == 2)
assert len(l) - 3 >= 0
