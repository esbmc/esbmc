d = {1: ('A', 'B'), 2: ('C', 'D')}
n = 0
for k, v in d.items():
    if k == 1:
        assert v[0] == 'A'
    if k == 2:
        assert v[1] == 'C'
    n += 1
assert n == 2
