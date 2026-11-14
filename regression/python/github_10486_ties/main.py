assert type(max(1, 1.0)) is int
assert type(max(1.0, 1)) is float
assert type(min(1, 1.0)) is int
assert type(min(1.0, 1)) is float
m = max(9007199254740993, 9007199254740992.0)
assert type(m) is int and m == 9007199254740993
assert type(min(9007199254740993, 9007199254740992.0)) is float
assert type(max(-0.5, 0)) is int
assert type(min(-0.5, 0)) is float
