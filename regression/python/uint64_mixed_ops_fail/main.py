class uint64(int):
    pass


big = uint64(2**63 + 5)
l = [1, 2, 3]
assert len(l) + big < 0
