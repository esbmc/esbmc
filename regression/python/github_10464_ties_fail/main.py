# 10464: 2**53 + 1 is greater than the float 2.0**53.
m = max([9007199254740993, 9007199254740992.0])
assert type(m) is float
