class uint256(int):
    pass


n = uint256(2**255 + 5)
assert n // 3 < 0
