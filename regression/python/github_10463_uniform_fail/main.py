# 10463: a sorted int list holds ints.
a = [3]
a.append(1)
a.sort()
assert type(a[0]) is float
