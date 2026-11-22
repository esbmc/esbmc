# 10462: the ints of a mixed int/float literal are not typed as floats.
s = sorted([2, 1.5])
assert type(s[1]) is float
