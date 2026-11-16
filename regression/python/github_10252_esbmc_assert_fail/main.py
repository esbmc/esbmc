from esbmc import __ESBMC_assert


def g(a: int) -> int:
    return a // 2


__ESBMC_assert(False, "mine")
print(g(4))
