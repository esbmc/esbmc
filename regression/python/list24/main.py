# Initial list
a = [1, 2, 3]

# Extend list a with elements [4, 5, 6]
a.extend([4, 5, 6])

# Expected output: [1, 2, 3, 4, 5, 6]
assert a[0] == 1
assert a[1] == 2
assert a[2] == 3
assert a[3] == 4
assert a[4] == 5
assert a[5] == 6
