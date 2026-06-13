# Lexicographic ordering comparison of tuples (<, <=, >, >=). Tuples are modelled
# as structs, and the SMT backend has no struct ordering, so a tuple `>` tripped
# an is_signedbv assertion. The frontend now lowers tuple ordering to element-wise
# comparisons: (a0,a1,..) < (b0,b1,..) == a0<b0 or (a0==b0 and (a1<b1 or ...)).

# First element decides.
assert (1, 2) < (3, 4)
assert (3, 4) > (1, 2)

# Equal first element: the second decides (lexicographic).
assert (1, 2) < (1, 5)
assert (2, 9) > (2, 1)

# <= / >= including equality.
assert (2, 3) <= (2, 3)
assert (2, 3) >= (2, 3)
assert (1, 0) <= (1, 1)

# Three elements, and signed comparison.
assert (1, 2, 3) < (1, 2, 4)
assert (-1, 5) < (0, 0)

# Float components, and a mixed int/float pair (Python promotes int->float).
assert (1.0, 2.0) < (1.0, 9.0)
assert (1, 2.0) < (1, 3.0)
assert (1.0, 2) < (2, 1)

# Nested-tuple components compare recursively.
assert ((1, 2), 3) < ((1, 2), 4)
assert ((1, 3), 0) > ((1, 2), 9)

# Tuples of different arity: a proper prefix is the smaller tuple.
assert (1, 2) < (1, 2, 0)
assert (1, 2, 3) > (1, 2)

# The practical use: a manual sort driven by tuple comparison.
pairs = [(3, 1), (1, 2)]
if pairs[0] > pairs[1]:
    pairs[0], pairs[1] = pairs[1], pairs[0]
assert pairs[0][0] == 1
assert pairs[1][0] == 3
