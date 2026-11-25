# Issue #10467: a non-literal step on a list parameter was sliced with step 1.
def f(l: list[int], s: int) -> int:
    return len(l[::s])


k: int = nondet_int()
__ESBMC_assume(k == 2)
assert f([1, 2, 3, 4, 5], k) == 5
