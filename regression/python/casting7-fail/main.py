# Ordered comparisons between float and char (str) should raise TypeError in Python

f = float(65)
assert not (f < 'A')

f = float(66)
assert not (f > 'A')

f = float(65)
assert not (f <= 'A')

f = float(65)
assert not (f >= 'A')
