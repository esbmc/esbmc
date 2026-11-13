# 10464: the first of equal extremes wins, and ints compare exactly.
assert type(max([1, 1.0])) is int
assert type(max([1.0, 1])) is float
assert type(min([1, 1.0])) is int
assert type(min([1.0, 1])) is float

m = max([9007199254740993, 9007199254740992.0])
assert type(m) is int and m == 9007199254740993
k = min([9007199254740993, 9007199254740992.0])
assert type(k) is float
