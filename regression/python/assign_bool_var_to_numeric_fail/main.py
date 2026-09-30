# Assigning a bool variable to a float variable converts the value and leaves
# the bool variable's own type alone.
b = True
x = 1.5
x = b
assert not isinstance(b, bool)
assert x == 1
