# Symmetric direction of the straight-line retyping fix (#4774): a module
# global rebound from str to a numeric scalar must adopt the numeric value.
x = 'hello'
assert x == 'hello'

x = 42
assert x == 42
assert x * 2 == 84
