# tuple() over a string yields a tuple of single-character strings
# (s becomes ('a', 'b')). This was previously rejected with an explicit
# error; it is now modelled (see python/tuple_from_str). Kept as a
# regression that the constant-string case verifies.
s = tuple("ab")
assert s[0] == "a" and s[1] == "b" and len(s) == 2
