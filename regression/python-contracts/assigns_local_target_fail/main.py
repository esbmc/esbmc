def f(n: int) -> int:
    k: int = 0
    __ESBMC_assigns(k)
    k = n
    return k

f(1)
