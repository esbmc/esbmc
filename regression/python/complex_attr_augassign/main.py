# Tests .real/.imag after augmented assignment on complex values

# += complex
z1 = complex(1.0, 2.0)
z1 += complex(3.0, 4.0)
assert z1.real == 4.0
assert z1.imag == 6.0

# -= complex
z2 = complex(10.0, 8.0)
z2 -= complex(3.0, 2.0)
assert z2.real == 7.0
assert z2.imag == 6.0

# *= complex
z3 = complex(1.0, 1.0)
z3 *= complex(1.0, 1.0)
# (1+1j)*(1+1j) = 0+2j
assert z3.real == 0.0
assert z3.imag == 2.0

# += float
z4 = complex(2.0, 3.0)
z4 += 5.0
assert z4.real == 7.0
assert z4.imag == 3.0

# *= float
z5 = complex(3.0, 4.0)
z5 *= 2.0
assert z5.real == 6.0
assert z5.imag == 8.0

# -= float
z6 = complex(10.0, 5.0)
z6 -= 3.0
assert z6.real == 7.0
assert z6.imag == 5.0

# += with conjugate result
z7 = complex(3.0, 4.0)
c7 = z7.conjugate()
z7 += c7
# (3+4j) + (3-4j) = 6+0j
assert z7.real == 6.0
assert z7.imag == 0.0
