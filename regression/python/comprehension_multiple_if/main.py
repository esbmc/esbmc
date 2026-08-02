# A comprehension may have several `if` clauses in one generator
# ([x for x in A if c1 if c2]), which is equivalent to conjoining them
# (if c1 and c2). ESBMC previously rejected this with NotImplementedError.
# List and set comprehensions now conjoin the conditions.
r = [x for x in range(6) if x > 1 if x < 5]
assert r == [2, 3, 4]

r3 = [x for x in range(8) if x > 0 if x % 2 == 0 if x < 7]
assert r3 == [2, 4, 6]

s = {x for x in range(6) if x > 2 if x < 5}
assert s == {3, 4}

# Single-if and `and` forms are unchanged.
assert [x for x in range(6) if x % 2 == 0] == [0, 2, 4]
assert [x for x in range(6) if x > 1 and x < 5] == [2, 3, 4]
