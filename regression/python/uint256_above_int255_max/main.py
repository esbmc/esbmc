class uint256(int):
    pass


n = uint256(2**255 + 5)
l = [1, 2, 3]
assert n >> 255 == 1
assert n / -2 < 0
assert n - len(l) > -1
