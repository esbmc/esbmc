# Buggy on purpose: assumes // truncates toward zero, as in C. Python's // floors.
def half(n: int) -> int:
    return n // 2


def check(n: int) -> bool:
    __ESBMC_assume(-1000 <= n and n <= 1000)
    assert half(-n) == -half(n)
    return True
