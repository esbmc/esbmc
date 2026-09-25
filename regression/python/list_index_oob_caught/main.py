# An out-of-bounds list index raises a *catchable* IndexError, matching
# CPython. Previously a non-negative out-of-bounds read tripped an internal
# "out-of-bounds read in list" assertion that try/except could not intercept,
# so this program wrongly reported VERIFICATION FAILED. The runtime bounds
# guard in build_list_at_call now raises IndexError for any out-of-bounds
# normalized index (not only negative ones).
x: list[int] = [1, 2, 3]
i: int = nondet_int()
__ESBMC_assume(i == 5)

caught: bool = False
try:
    v: int = x[i]
except IndexError:
    caught = True

assert caught
