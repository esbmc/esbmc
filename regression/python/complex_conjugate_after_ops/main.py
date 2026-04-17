# Tests conjugate on complex results of arithmetic operations

# Conjugate of a sum
a = complex(1.0, 2.0)
b = complex(3.0, 4.0)
cs = (a + b).conjugate()
assert cs.real == 4.0
assert cs.imag == -6.0

# Conjugate of a product
cp = (a * b).conjugate()
# a*b = (1+2j)(3+4j) = -5+10j => conj = -5-10j
assert cp.real == -5.0
assert cp.imag == -10.0

# Conjugate of a difference
cd = (b - a).conjugate()
assert cd.real == 2.0
assert cd.imag == -2.0

# Conjugate of a division
z1 = complex(4.0, 2.0)
z2 = complex(1.0, 1.0)
cdiv = (z1 / z2).conjugate()
# z1/z2 = 3-1j => conj = 3+1j
assert cdiv.real == 3.0
assert cdiv.imag == 1.0

# Conjugate of negation
z3 = complex(3.0, 4.0)
neg_z3 = -z3
cn = neg_z3.conjugate()
assert cn.real == -3.0
assert cn.imag == 4.0
