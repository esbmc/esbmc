d = {1: [(1, 2), (3, 4)], 2: [(5, 6)]}
a = d[1]
t = a[1]
assert t[0] == 4
assert t[1] == 4
b = d[2]
u = b[0]
assert u[1] == 6

nums = {1: [10, 20, 30]}
lst = nums[1]
assert lst[1] == 20
assert len(lst) == 3
