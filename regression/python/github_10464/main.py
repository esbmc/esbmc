# 10464: min() and max() return the selected element with its own type.
assert type(max([1, 0.5])) is int
assert type(min([1, 0.5])) is float
assert max([1, 0.5]) == 1 and min([1, 0.5]) == 0.5

n = min([-2, -1.5, 3])
assert n == -2 and type(n) is int
