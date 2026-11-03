# #8100: appends through a parameter or an alias must reach list ==.
m: list = [1]
n: list = [1]
a = m
b = n
a.append(1)
b.append(2)
assert m == n
