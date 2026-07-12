# Positive companion of github_5915_cf_fail: pinning c <= 0 forces n == 2, so
# (n + 0) ** 2 is exactly 4. The old fold to the stale value 3 gave 9 and
# reported this as violated; a sound symbolic base makes it hold.
c = nondet_int()
n = 2
if c > 0:
    n = 3
__ESBMC_assume(c <= 0)
assert (n + 0) ** 2 == 4
