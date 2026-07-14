# Negative companion: the nested unpacking is performed correctly, but the
# asserted sum is wrong, so verification must FAIL.
items = [((1, 2), 3), ((4, 5), 6)]
total = 0
for (a, b), c in items:
    total = total + a + b + c
assert total == 99
