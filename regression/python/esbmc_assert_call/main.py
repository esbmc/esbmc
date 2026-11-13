def positive(v: int) -> bool:
    return v > 0


n = nondet_int()
if n > 0:
    __ESBMC_assert(positive(n), "n is positive")
