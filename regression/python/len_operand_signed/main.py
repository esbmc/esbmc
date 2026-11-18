def four() -> int:
    return 4


l = [1, 2, 3]
m = [1, 2, 3, 4]
assert len(l) - len(m) < 0
assert len(l) - four() < 0
assert (n := len(l)) - 4 < 0
