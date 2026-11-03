# GitHub #9054: an index is an int, so only the second assertion fails.
s = "abc"
z = s.index("a")
assert isinstance(z, int)
assert isinstance(z, str)
