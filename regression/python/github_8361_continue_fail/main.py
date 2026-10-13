i = 0
odd = 0
while (j := i) < 5:
    i += 1
    if j % 2 == 0:
        continue
    odd += 1
assert j == 4
