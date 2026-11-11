# Issue #10228: 2 ** n with a non-constant int n passed to range()
n = len([1, 2, 3])
count = 0
for mask in range(2 ** n):
    count += 1
assert count == 8
