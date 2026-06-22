# Negative: the nested-key unpack is real, so a wrong expectation must FAIL.
d = {('A', 'B'): 3, ('C', 'D'): 7}
total = 0
for (u, v), w in d.items():
    total += w
assert total == 99
