# Negative companion to tuple-from-list: tuples built from lists with
# different contents must compare unequal, so this assertion fails.
a = tuple([1, 2, 3])
b = tuple([1, 2, 4])
assert a == b
