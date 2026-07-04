# The even values of range(5) sum to 6 (0+2+4), not 5.
s = 0
for i in range(5):
    if i % 2:
        continue
    s += i
assert s == 5
