# GitHub #9054: an index is an int, so only the second assertion fails.
x = ["a"]
z = x.index("a")
assert isinstance(z, int)
assert isinstance(z, str)
