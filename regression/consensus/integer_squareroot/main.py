class uint64(int):
    pass


UINT64_MAX = uint64(2**64 - 1)


UINT64_MAX_SQRT = uint64(4294967295)


def integer_squareroot(n: uint64) -> uint64:
    """
    Return the largest integer ``x`` such that ``x**2 <= n``.
    """
    if n == UINT64_MAX:
        return UINT64_MAX_SQRT
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

