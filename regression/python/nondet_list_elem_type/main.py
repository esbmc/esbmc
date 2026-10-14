# The element type must reach the elements: an int list cannot hold 0.5.
# A polymorphic builder that dispatched on a type tag degraded every element to
# the first branch's type, which is why models/nondet.py is monomorphic.
x: list[int] = nondet_list(3, nondet_int())
__ESBMC_assume(len(x) >= 1)
assert x[0] != 0.5
