# Ints, where the comprehension variable used to give the later dict an int type.
e = [i + 1 for i in [1, 2]]
i = {1: 2.0}
assert i[1] == 2.0
