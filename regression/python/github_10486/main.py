assert type(max(1, 0.5)) is int
assert type(min(1, 0.5)) is float
assert max(1, 0.5) == 1 and min(1, 0.5) == 0.5
assert type(max(-2, -2.5, -3)) is int
assert type(min(3, 2.5, -1)) is int
assert type(max(True, 0.5)) is bool
