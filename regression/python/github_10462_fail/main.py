# 10462: sorted() of a list literal keeps the literal's element types.
s = sorted([2.5, 1.0])
assert type(s[0]) is float
assert s[0] == 1
assert s[1] == 1
