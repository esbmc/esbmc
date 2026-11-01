# The shape of the Typpete task with a list: two comprehension targets, then a rebinding.
d = [[1.5, 2.5], [3.5]]
e = [(i, i * 2) for j in d for i in j]
i, j = {1: 2.0}, True
assert i[1] == 2.0
assert j
assert len(e) == 3
