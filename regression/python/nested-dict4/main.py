d1: dict[int, dict[int, int]] = {
    10: {1: 2},
    20: {3: 4},
}
assert d1[20][3] == 4

d2: dict[int, dict[int, int]] = {
    1: {999999: 42},
}
assert d2[1][999999] == 42

