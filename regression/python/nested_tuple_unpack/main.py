# Nested tuple unpacking targets, e.g. `(a, b), c`, must be supported wherever
# the RHS is a runtime value (not just a literal flattened at compile time):
# for-loop iteration, direct assignment from a subscript, and function returns.
items = [((1, 2), 3), ((4, 5), 6)]
total = 0
for (a, b), c in items:
    total = total + a + b + c
assert total == 21

# Two nested tuples in one target.
data = [((1, 2), (3, 4))]
s = 0
for (a, b), (c, d) in data:
    s = s + a + b + c + d
assert s == 10

# Nested unpack from a function return value.
def pair_and_value():
    return (7, 8), 9

(x, y), z = pair_and_value()
assert x == 7 and y == 8 and z == 9
