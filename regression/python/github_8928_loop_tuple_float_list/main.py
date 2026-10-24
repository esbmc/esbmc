for (r, m) in [([1.0], 3.0)]:
    assert r[0] * m == 3.0

for (r, m) in [([0.5, 2.0], 4.0), ([1.5], 2.0)]:
    assert r[0] * m == 2.0 or r[0] * m == 3.0

q, k = ([2.5], 2.0)
assert q[0] * k == 5.0
