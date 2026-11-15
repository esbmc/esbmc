class uint64(int):
    pass


big = uint64(2**63 + 5)
assert big // 3 == 3074457345618258604
assert big % 7 == 6
assert big >> 63 == 1
assert big >> uint64(63) == 1
assert len([1, 2]) + big > big
