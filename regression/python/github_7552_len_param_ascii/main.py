def g(s: str) -> int:
    return len(s)


n = nondet_int()
__ESBMC_assume(0 <= n and n <= 2)
assert g("abc"[n:]) == 3 - n
