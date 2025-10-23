# Initial lists
a = [1, 2, 3]
b = [4, 5, 6]

# Extend list a with elements from list b
a.extend(b)

# Expected output: [1, 2, 3, 4, 5, 6]
assert a[0] == 1
assert a[1] == 2
assert a[2] == 3
assert a[3] == 4
assert a[4] == 5
assert a[5] == 6
