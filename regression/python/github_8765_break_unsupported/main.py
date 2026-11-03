ts = (str, int)
n = 0
for a in ts:
    if a is int:
        n += 1
        break
assert n == 1
