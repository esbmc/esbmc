# Guard: set union (|) and integer bitwise-or must still work after the dict-|
# diagnostic was added to the same BitOr path.
s = {1, 2} | {3, 4}
assert 3 in s
assert 1 in s
x = 6 | 1
assert x == 7
