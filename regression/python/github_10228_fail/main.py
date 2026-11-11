# Issue #10228
n = len([1, 2, 3])
count = 0
for mask in range(2 ** n):
    count += 1
assert count == 7
