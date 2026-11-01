# Issue #9359: 2**53 + 1 is not equal to 2.0**53.
a = 9007199254740993
b = 9007199254740992.0
assert a == b
