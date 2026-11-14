def mid(lo: int, hi: int) -> int:
    return (lo + hi) // 2


def check(lo: int, hi: int) -> bool:
    __ESBMC_assume(lo <= hi)
    m: int = mid(lo, hi)
    assert lo <= m and m <= hi
    return True
