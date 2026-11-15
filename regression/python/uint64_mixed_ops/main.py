class uint64(int):
    pass


big = uint64(2**63 + 5)
l = [1, 2, 3]
assert big // 3 == 3074457345618258604
assert big % 7 == 6
assert big >> 63 == 1
assert big >> uint64(63) == 1
assert len(l) + big > -1
q = uint64(9) % 4
xs: list[int] = [q]
assert xs[0] == 1
m = uint64(8) >> True
ys: list[int] = [m]
assert ys[0] == 4
assert big / -2 < 0
