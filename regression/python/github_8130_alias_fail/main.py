# #8130: replacing an inner list through an alias must reach list ==.
m: list = [[1]]
n: list = [[1]]
k = m
k[0] = [5]
assert m == n
