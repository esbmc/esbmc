# Issue #10494: len(x) - k was computed unsigned and wrapped.
s = nondet_str()
__ESBMC_assume(len(s) == 2)
assert len(s) - 3 < 0
l = nondet_list()
__ESBMC_assume(len(l) == 2)
assert len(l) - 3 < 0
