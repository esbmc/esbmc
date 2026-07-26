# {0: 1, 1: 2.5}[1] is 2.5, not 2 (the old truncated/misread value).
d = {0: 1, 1: 2.5}
assert d[1] == 2
