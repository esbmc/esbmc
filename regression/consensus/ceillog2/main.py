class uint64(int):
    pass


def ceillog2(x: int) -> uint64:
    if x < 1:
        raise ValueError(f"ceillog2 accepts only positive values, x={x}")
    return uint64((x - 1).bit_length())


x = nondet_int()
__ESBMC_assume(x >= 1 and x <= 64)
result = ceillog2(x)
assert (1 << result) >= x
assert result == 0 or x > (1 << (result - 1))
