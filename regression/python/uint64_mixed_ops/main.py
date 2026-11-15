class uint64(int):
    pass


big = uint64(2**63 + 5)
seven = uint64(7)
assert seven // -2 == -4
assert seven % -2 == -1
assert big >> 63 == 1
assert big >> uint64(63) == 1
assert len([1, 2]) + big > big
