d1: dict[int, dict[int, int]] = {1: {1: 10}}
assert d1[1][1] == 10

d2: dict[int, dict[int, int]] = {1: {100: 20}}
assert d2[1][100] == 20

d3: dict[int, dict[int, int]] = {1: {-5: 30}}
assert d3[1][-5] == 30

d4: dict[int, dict[int, int]] = {0: {3: 4}}
assert d4[0][3] == 4

d5: dict[int, dict[int, int]] = {99: {3: 4}}
assert d5[99][3] == 4

d6: dict[int, dict[int, int]] = {-7: {3: 4}}
assert d6[-7][3] == 4

d7: dict[int, dict[int, int]] = {1: {3: 4, 10: 99, -5: -7}}
assert d7[1][10] == 99
