# Soundness guard for the symbolic-range dict-comprehension fix (#5222): a
# wrong value must still FAIL. d[2] is 7, not 99.
n = 3
d = {i: 7 for i in range(n)}
assert d[2] == 99
