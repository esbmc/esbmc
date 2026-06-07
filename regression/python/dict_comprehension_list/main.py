# Dict comprehension whose iterable is a list (not range). The loop variable
# must be typed with the list's element type so each key is boxed with an
# aligned representation; otherwise a later lookup trips a dereference-alignment
# failure in __ESBMC_values_equal. Regression for the dict-comprehension key
# storage bug.
nums = [1, 2, 3]
squares = {n: n * n for n in nums}
assert squares[2] == 4
assert squares[3] == 9

vals = [1.5, 2.5]
plus_one = {v: v + 1.0 for v in vals}
assert plus_one[1.5] == 2.5

# Comprehension with a filter over a list iterable.
evens = {x: x for x in nums if x % 2 == 0}
assert evens[2] == 2
