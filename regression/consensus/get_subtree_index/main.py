class GeneralizedIndex(int):
    pass


class uint64(int):
    pass


def floorlog2(x: int) -> uint64:
    if x < 1:
        raise ValueError(f"floorlog2 accepts only positive values, x={x}")
    return uint64(x.bit_length() - 1)


def get_subtree_index(generalized_index: GeneralizedIndex) -> uint64:
    return uint64(generalized_index % 2**(floorlog2(generalized_index)))


generalized_index = nondet_int()
__ESBMC_assume(generalized_index >= 1 and generalized_index <= 64)
result = get_subtree_index(generalized_index)
depth = floorlog2(generalized_index)
assert result == generalized_index - (1 << depth)
