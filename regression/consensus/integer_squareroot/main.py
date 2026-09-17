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
    # Unwinding this loop over the full uint64 domain doesn't converge in
    # reasonable time (the division makes each extra unwind expensive), so
    # test.desc proves it via --loop-invariant-check instead: x and y stay
    # >= 1 whenever n >= 1, so n // x never divides by zero.
    __loop_invariant((n == 0 and x == 0 and y == 0) or
                      (n >= 1 and x >= 1 and y >= 1))
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

