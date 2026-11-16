i, j, (k, (l, m)) = {1: 2.0}, True, ((1, 2), (3, {4}))
for key, value in i.items():
    assert isinstance(key, int)
    assert isinstance(value, float)
assert j
assert k == (1, 2)
assert l == 3
