class uint64(int):
    pass


big = uint64(2**63 + 5)
r = -1 % big
assert r < 2**62
