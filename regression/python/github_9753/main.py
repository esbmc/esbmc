a = 1
c = {1: 1 + 1, 2: a, 3: int(3)}
total = 0
for v in c.values():
    assert isinstance(v, int)
    total += v
assert total == 6
for k, v in c.items():
    assert isinstance(v, int)
assert isinstance(list(c.values())[0], int)
d = {1: True and False}
for b in d.values():
    assert isinstance(b, bool)
    assert not b
