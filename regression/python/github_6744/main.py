pairs = [(1, 2), (3, 4)]
total = 0
for i, (a, b) in enumerate(pairs):
    total += i + a + b
assert total == 11

triples = 0
for j, (x, y, z) in enumerate([(1, 2, 3), (4, 5, 6)], 1):
    triples += j * (x + y + z)
assert triples == 36

shifted = 0
for k, (p, q) in enumerate(pairs, -1):
    shifted += k * (p + q)
assert shifted == -3

empty: list = []
for m, (u, v) in enumerate(empty):
    assert False
