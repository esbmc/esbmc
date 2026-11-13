# 10464: the issue's program, max([1, 0.5]) is the int 1.
assert type(max([1, 0.5])) is float
