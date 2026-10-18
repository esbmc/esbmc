items = [1, 2, 3]
total = 0
for x in items:
    total = total + x
assert total == 6

letters = 0
for ch in "abc":
    letters = letters + 1
assert letters == 3

assert 2 in items
assert 7 not in items
