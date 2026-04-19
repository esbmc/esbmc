x = None
assert x is None
assert x != 0
assert x != False
assert x != ''
assert x != []
assert x is not True

# Identity and equality edge cases
y = None
assert x is y
assert not (x is not y)
assert (x == y)
assert (x is None) and (y is None)

# Comparison operations that must not raise
assert (x == None)
assert not (x != None)
assert not (x == 0)
assert not (x == False)
assert (x is None)

# Check None inside logical expressions
assert not (x and True)
assert (x or True)
assert (x or 2) == 2
assert not (x and 1)
