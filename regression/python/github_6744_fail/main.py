pairs = [(1, 2), (3, 4)]
total = 0
for i, (a, b) in enumerate(pairs):
    total += i + a + b
assert total == 10
