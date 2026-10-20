# The list held in p has no recorded element types, so x and y cannot be
# typed exactly and the unpacking is refused.
p = ([1, 2], 3)
(x, y), m = p
assert x * m == 3
