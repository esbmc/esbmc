# A slice with step 0 raises a catchable ValueError (previously an uncatchable
# property violation). Covers list and string slices.
try:
    [1, 2, 3][::0]
    assert False
except ValueError:
    pass

try:
    "abc"[::0]
    assert False
except ValueError:
    pass

l = [1, 2, 3, 4]
try:
    l[1::0]
    assert False
except ValueError:
    pass

# Valid slices are unchanged.
assert [0, 1, 2, 3, 4][::2] == [0, 2, 4]
assert [1, 2, 3][::-1] == [3, 2, 1]
