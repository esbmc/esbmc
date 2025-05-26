x = 0
for y in range(1, 5, 0):  # Zero step - should fail
    x = y
assert x == 0
