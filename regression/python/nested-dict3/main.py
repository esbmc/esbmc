d1: dict[int, dict[int, int]] = {
    1: {3: 4},
    2: {5: 6},
    7: {8: 9},
}
assert d1[7][8] == 9

d2: dict[int, dict[int, int]] = {1: {3: -1}}
assert d2[1][3] == -1

d3: dict[int, dict[int, int]] = {1: {3: 987654321}}
assert d3[1][3] == 987654321

d4: dict[int, dict[int, int]] = {
    1000: {2000: 3000},
    -1000: {-2000: -3000},
}
assert d4[-1000][-2000] == -3000

