class uint64(int):
    pass


UINT64_MAX = uint64(2**64 - 1)


UINT64_MAX_SQRT = uint64(4294967295)


# test.desc uses --unwind 8 --no-unwinding-assertions rather than
# --incremental-bmc: this loop's integer division makes each extra unwind
# expensive, and --incremental-bmc keeps growing the bound trying to prove
# completeness instead of just checking one -- it doesn't converge in
# reasonable time. The bound is sound only up to n that converge within
# 8 iterations.
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

