# Issue #9386: a negative shift count raises ValueError.
def scale(v: int, k: int) -> int:
    return v << k


k: int = nondet_int()
__ESBMC_assume(k >= -4 and k <= 4)
r: int = scale(1, k)
