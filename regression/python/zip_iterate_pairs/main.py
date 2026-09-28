# The zip shape that already works: iterating the pairs, including the
# truncation to the shorter input. Pinned so a list(zip(...)) fix cannot
# regress it.

n = 0
for a, b in zip([1, 2, 3], [4, 5]):
    n += 1
assert n == 2

total = 0
for a, b in zip([1, 2], [10, 20]):
    total += a * b
assert total == 50
