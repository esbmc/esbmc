class uint64(int):
    pass


def floorlog2(x: int) -> uint64:
    if x < 1:
        raise ValueError(f"floorlog2 accepts only positive values, x={x}")
    return uint64(x.bit_length() - 1)


x = nondet_int()
__ESBMC_assume(x >= 1 and x <= 64)
result = floorlog2(x)
assert (1 << result) <= x
assert x < (1 << (result + 1))
