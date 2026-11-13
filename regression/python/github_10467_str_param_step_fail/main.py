def pick(w: str, k: int) -> str:
    return w[::k]


k = nondet_int()
__ESBMC_assume(k == 2 or k == 3)
r = pick("abcdef", k)
assert r == "ace"
