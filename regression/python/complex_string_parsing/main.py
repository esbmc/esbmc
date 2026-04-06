# Tests for complex() string parsing via parse_complex_string utility.
# Validates all CPython-accepted string formats are correctly handled.

# Pure real strings
z1 = complex("0")
assert z1.real == 0.0 and z1.imag == 0.0

z2 = complex("42")
assert z2.real == 42.0 and z2.imag == 0.0

z3 = complex("-3.5")
assert z3.real == -3.5 and z3.imag == 0.0

z4 = complex("1e2")
assert z4.real == 100.0 and z4.imag == 0.0

# Pure imaginary strings
z5 = complex("j")
assert z5.real == 0.0 and z5.imag == 1.0

z6 = complex("+j")
assert z6.real == 0.0 and z6.imag == 1.0

z7 = complex("-j")
assert z7.real == 0.0 and z7.imag == -1.0

z8 = complex("2j")
assert z8.real == 0.0 and z8.imag == 2.0

z9 = complex("-3.5j")
assert z9.real == 0.0 and z9.imag == -3.5

# Combined real+imag strings
z10 = complex("1+2j")
assert z10.real == 1.0 and z10.imag == 2.0

z11 = complex("1-2j")
assert z11.real == 1.0 and z11.imag == -2.0

z12 = complex("-1+2j")
assert z12.real == -1.0 and z12.imag == 2.0

z13 = complex("-1-2j")
assert z13.real == -1.0 and z13.imag == -2.0

# Unit imaginary with real part
z14 = complex("3+j")
assert z14.real == 3.0 and z14.imag == 1.0

z15 = complex("3-j")
assert z15.real == 3.0 and z15.imag == -1.0

# Scientific notation
z16 = complex("1e3+2e-1j")
assert z16.real == 1000.0 and z16.imag == 0.2

z17 = complex("1.5E2-3.0E1j")
assert z17.real == 150.0 and z17.imag == -30.0

# Parenthesized form
z18 = complex("(3+4j)")
assert z18.real == 3.0 and z18.imag == 4.0

# Whitespace-trimmed
z19 = complex("  5  ")
assert z19.real == 5.0 and z19.imag == 0.0

z20 = complex(" (3+4j) ")
assert z20.real == 3.0 and z20.imag == 4.0

# Malformed strings -> ValueError
raised = False
try:
    complex("x")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1 + 2j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("++1j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("")
except ValueError:
    raised = True
assert raised

# Zero imaginary explicit
z21 = complex("0j")
assert z21.real == 0.0 and z21.imag == 0.0

# Decimal without integer part
z22 = complex(".5j")
assert z22.real == 0.0 and z22.imag == 0.5

# Negative zero IEEE
z24 = complex("-0.0")
assert z24.imag == 0.0
