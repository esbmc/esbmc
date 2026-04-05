# Tests for complex() constructor keyword argument handling.
# Validates that keyword arguments are correctly parsed and
# error conditions are properly detected.

# Named keyword: real only
z1 = complex(real=3.5)
assert z1.real == 3.5
assert z1.imag == 0.0

# Named keyword: imag only
z2 = complex(imag=4.0)
assert z2.real == 0.0
assert z2.imag == 4.0

# Named keywords: both
z3 = complex(real=1.0, imag=2.0)
assert z3.real == 1.0
assert z3.imag == 2.0

# Named keywords: reversed order
z4 = complex(imag=2.0, real=1.0)
assert z4.real == 1.0
assert z4.imag == 2.0

# Named keyword with complex value
c1 = complex(1.0, 2.0)
z5 = complex(real=c1, imag=3.0)
assert z5.real == 1.0
assert z5.imag == 5.0

# Named keyword: string in real (one-arg form equivalent)
z6 = complex(real="5+6j")
assert z6.real == 5.0
assert z6.imag == 6.0

# Unexpected keyword -> TypeError
raised_type = False
try:
    complex(foo=1)
except TypeError:
    raised_type = True
assert raised_type

# Positional + duplicate keyword 'real' -> TypeError
raised_type = False
try:
    complex(1, real=2)
except TypeError:
    raised_type = True
assert raised_type

# Positional + duplicate keyword 'imag' -> TypeError
raised_type = False
try:
    complex(1, 2, imag=3)
except TypeError:
    raised_type = True
assert raised_type

# Integer keyword value for real
z7 = complex(real=0)
assert z7.real == 0.0
assert z7.imag == 0.0

# Bool keyword value for imag
z8 = complex(imag=True)
assert z8.real == 0.0
assert z8.imag == 1.0
