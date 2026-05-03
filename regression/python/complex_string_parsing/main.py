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

# Decimal without integer part (leading dot)
z25 = complex(".5")
assert z25.real == 0.5 and z25.imag == 0.0

z26 = complex(".5+.3j")
assert z26.real == 0.5 and z26.imag == 0.3

# More scientific notation edge cases
z27 = complex("-1e-2-3e-1j")
assert z27.real == -0.01 and z27.imag == -0.3

z28 = complex("2.5e1+1.5e-1j")
assert z28.real == 25.0 and z28.imag == 0.15

# Nested parentheses are NOT accepted by CPython -> ValueError
raised = False
try:
    complex("((3+4j))")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("(((1+1j)))")
except ValueError:
    raised = True
assert raised

# Pure real with E notation
z31 = complex("1E10")
assert z31.real == 1.0e10 and z31.imag == 0.0

z32 = complex("2.5E-3")
assert z32.real == 0.0025 and z32.imag == 0.0

# Pure imaginary with E notation
z33 = complex("1E5j")
assert z33.real == 0.0 and z33.imag == 1.0e5

z34 = complex("3.5E-2j")
assert z34.real == 0.0 and z34.imag == 0.035

# Positive sign explicit
z35 = complex("+5")
assert z35.real == 5.0 and z35.imag == 0.0

z36 = complex("+3+4j")
assert z36.real == 3.0 and z36.imag == 4.0

# Mixed signs with E notation
z37 = complex("-2E3+1E-1j")
assert z37.real == -2000.0 and z37.imag == 0.1

z38 = complex("1.0E2-2.5E1j")
assert z38.real == 100.0 and z38.imag == -25.0

# Zero cases
z39 = complex("0.0")
assert z39.real == 0.0 and z39.imag == 0.0

z40 = complex("0+0j")
assert z40.real == 0.0 and z40.imag == 0.0

z41 = complex("+0-0j")
assert z41.real == 0.0 and z41.imag == 0.0

# ====== ERROR/EXCEPTION CASES ======

# Invalid suffixes
raised = False
try:
    complex("1+2k")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("3x")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("2i")  # Should be 'j', not 'i'
except ValueError:
    raised = True
assert raised

# Double operators
raised = False
try:
    complex("--1j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1--2j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1++2j")
except ValueError:
    raised = True
assert raised

# Invalid digits
raised = False
try:
    complex("1a+2j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1+2aj")
except ValueError:
    raised = True
assert raised

# Incomplete expressions
raised = False
try:
    complex("1+")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("+j+2")
except ValueError:
    raised = True
assert raised

# Whitespace-only (empty string already covered above)
raised = False
try:
    complex("   ")
except ValueError:
    raised = True
assert raised

# Unbalanced parentheses
raised = False
try:
    complex("(1+2j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1+2j)")
except ValueError:
    raised = True
assert raised

# Invalid decimal format
raised = False
try:
    complex("..5j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1.2.3j")
except ValueError:
    raised = True
assert raised

# Spaces inside parentheses (bare "1 + 2j" already covered above)
raised = False
try:
    complex("( 1 + 2j )")
except ValueError:
    raised = True
assert raised

# Multiple j suffixes
raised = False
try:
    complex("1j+2j")
except ValueError:
    raised = True
assert raised

# Mixed valid/invalid parts
raised = False
try:
    complex("1+2j3")
except ValueError:
    raised = True
assert raised

# ====== ADDITIONAL EDGE CASES ======

# Capital 'J' is also accepted as imaginary unit (CPython parity)
zJ1 = complex("1+2J")
assert zJ1.real == 1.0 and zJ1.imag == 2.0

zJ2 = complex("3J")
assert zJ2.real == 0.0 and zJ2.imag == 3.0

zJ3 = complex("J")
assert zJ3.real == 0.0 and zJ3.imag == 1.0

zJ4 = complex("+J")
assert zJ4.real == 0.0 and zJ4.imag == 1.0

zJ5 = complex("-J")
assert zJ5.real == 0.0 and zJ5.imag == -1.0

# Tabs and newlines count as whitespace and are trimmed
zW1 = complex("\t1+2j\t")
assert zW1.real == 1.0 and zW1.imag == 2.0

zW2 = complex("\n5\n")
assert zW2.real == 5.0 and zW2.imag == 0.0

zW3 = complex(" \t (3+4j) \t ")
assert zW3.real == 3.0 and zW3.imag == 4.0

# Internal whitespace around the sign is rejected
raised = False
try:
    complex("+ j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("- j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1+ 2j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1 +2j")
except ValueError:
    raised = True
assert raised

# Lone tokens / incomplete numbers are rejected
raised = False
try:
    complex("()")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("(")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex(")")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("+")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("-")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex(".")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex(".j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("e1")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1e")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1e+")
except ValueError:
    raised = True
assert raised

# Underscores inside numeric literals are not accepted by this parser
raised = False
try:
    complex("_1")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1_")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("1__0")
except ValueError:
    raised = True
assert raised

# Signed-zero imaginary forms are accepted
zS1 = complex("+0j")
assert zS1.real == 0.0 and zS1.imag == 0.0

zS2 = complex("-0j")
assert zS2.real == 0.0 and zS2.imag == 0.0

# Signed leading dot in real and imaginary parts
zS3 = complex("+.5j")
assert zS3.real == 0.0 and zS3.imag == 0.5

zS4 = complex("-.5+.5j")
assert zS4.real == -0.5 and zS4.imag == 0.5

# Explicit-sign exponents (e+/e-) on real and imaginary parts
zE1 = complex("1e+2")
assert zE1.real == 100.0 and zE1.imag == 0.0

zE2 = complex("1.5e+10")
assert zE2.real == 1.5e10 and zE2.imag == 0.0

zE3 = complex("1E+0")
assert zE3.real == 1.0 and zE3.imag == 0.0

zE4 = complex("0e0")
assert zE4.real == 0.0 and zE4.imag == 0.0

zE5 = complex("1e0j")
assert zE5.real == 0.0 and zE5.imag == 1.0

# Combined real+imag with explicit-sign exponents
zE6 = complex("1e2+3e2j")
assert zE6.real == 100.0 and zE6.imag == 300.0

zE7 = complex("1.5e-1-2.5e+1j")
assert zE7.real == 0.15 and zE7.imag == -25.0

# Trailing dot with exponent
zE8 = complex("5.e2")
assert zE8.real == 500.0 and zE8.imag == 0.0

zE9 = complex("5.e2j")
assert zE9.real == 0.0 and zE9.imag == 500.0

# Very large / very small magnitudes near double range
zE10 = complex("1e308")
assert zE10.real == 1e308 and zE10.imag == 0.0

zE11 = complex("1e-308")
assert zE11.real == 1e-308 and zE11.imag == 0.0

# Imaginary unit wrapped in parentheses
zP1 = complex("(j)")
assert zP1.real == 0.0 and zP1.imag == 1.0

zP2 = complex("(-j)")
assert zP2.real == 0.0 and zP2.imag == -1.0

zP3 = complex("(+j)")
assert zP3.real == 0.0 and zP3.imag == 1.0

# Parentheses with whitespace only are rejected
raised = False
try:
    complex("( )")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("(  )")
except ValueError:
    raised = True
assert raised

# Identifier-like strings are rejected
raised = False
try:
    complex("True")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("False")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("None")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("abc")
except ValueError:
    raised = True
assert raised

# Double signs / sign with lone dot are rejected
raised = False
try:
    complex("++")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("--")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("+-")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("-+")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("+.")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("-.")
except ValueError:
    raised = True
assert raised

# Sign + lone dot + j is rejected
raised = False
try:
    complex("+.j")
except ValueError:
    raised = True
assert raised

raised = False
try:
    complex("-.j")
except ValueError:
    raised = True
assert raised

# ====== CONTROL-FLOW SCENARIOS (if / while) ======

# 'if' branching on the parsed real/imag values
zif = complex("3+4j")
if zif.real > 0 and zif.imag > 0:
    assert zif.real == 3.0
    assert zif.imag == 4.0
else:
    assert False  # unreachable

zif2 = complex("-2.5")
if zif2.imag == 0.0:
    assert zif2.real == -2.5
else:
    assert False

# 'if' chain selecting the conjugate / negation manually
zif3 = complex("1-1j")
conj_real = zif3.real
conj_imag = zif3.imag
if zif3.imag < 0.0:
    conj_imag = 0.0 - zif3.imag
assert conj_real == 1.0 and conj_imag == 1.0

# 'while' accumulating parsed values (bounded loop ESBMC can unwind)
total_real = 0.0
total_imag = 0.0
i = 0
while i < 4:
    z = complex("1+1j")
    total_real = total_real + z.real
    total_imag = total_imag + z.imag
    i = i + 1
assert total_real == 4.0
assert total_imag == 4.0

# 'while' that exits early via break when the parsed value matches
i = 0
found_real = -1.0
while i < 5:
    z = complex("2+0j")
    if z.real == 2.0:
        found_real = z.real
        break
    i = i + 1
assert found_real == 2.0

# 'while' counting accepted ValueError cases (control-flow with exceptions)
bad_count = 0
j = 0
while j < 3:
    try:
        complex("bad")
    except ValueError:
        bad_count = bad_count + 1
    j = j + 1
assert bad_count == 3

# 'if' guarding a re-parse: parse, then validate via a second complex() call
zg = complex("(5+6j)")
if zg.real == 5.0:
    zg2 = complex("5+6j")  # without parens, should match
    assert zg2.real == zg.real
    assert zg2.imag == zg.imag
else:
    assert False
