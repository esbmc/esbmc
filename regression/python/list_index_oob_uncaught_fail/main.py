# The catchable IndexError must still fail verification when it is NOT caught:
# an uncaught out-of-bounds access propagates to the top-level uncaught-exception
# check and yields VERIFICATION FAILED. Guards against the runtime bounds guard
# silently dropping genuine out-of-bounds accesses.
x: list[int] = [1, 2, 3]
i: int = nondet_int()
__ESBMC_assume(i == 5)

v: int = x[i]
