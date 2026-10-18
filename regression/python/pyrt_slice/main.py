# Slicing follows CPython's normalisation: a negative bound counts from the
# end, one out of range clamps instead of raising, and a missing bound means
# the far end of whichever direction the step runs.
a = [1, 2, 3, 4, 5]
assert a[1:3] == [2, 3]
assert a[:2] == [1, 2]
assert a[3:] == [4, 5]
assert a[:] == [1, 2, 3, 4, 5]
assert a[-2:] == [4, 5]
assert a[:-3] == [1, 2]
assert a[1:4:2] == [2, 4]
assert a[::-1] == [5, 4, 3, 2, 1]
assert a[10:] == []

s = "abcde"
assert s[1:3] == "bc"
assert s[::-1] == "edcba"
assert s[:2] == "ab"
assert s[-1:] == "e"

t = (1, 2, 3)
assert t[1:] == (2, 3)
