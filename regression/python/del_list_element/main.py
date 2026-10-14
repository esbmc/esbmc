# `del a[i]` on a list removes and shifts out the element at index i (modelled
# as list.pop(i)). Covers constant, negative, and variable indices, deletion
# after an in-place mutation, and that dict deletion still works.
a = [1, 2, 3]
del a[1]
assert a[0] == 1 and a[1] == 3 and len(a) == 2

b = [10, 20, 30]
del b[-1]
assert b == [10, 20] and len(b) == 2

c = [1, 2, 3, 4]
i = 2
del c[i]
assert c[2] == 4 and len(c) == 3

e = [1, 2, 3]
e.append(4)
del e[1]
assert e == [1, 3, 4]

d = {1: 10, 2: 20}
del d[1]
assert 1 not in d and len(d) == 1
