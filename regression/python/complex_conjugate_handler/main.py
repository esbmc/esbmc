import math

# Basic conjugate.
z1 = complex(3, 4)
c1 = z1.conjugate()
assert c1 == complex(3, -4)

# Double conjugate returns original (using intermediate variable).
c1_back = c1.conjugate()
assert c1_back == z1

# Conjugate of purely real number.
z3 = complex(5, 0)
c3 = z3.conjugate()
assert c3 == z3

# Conjugate of purely imaginary number.
z4 = complex(0, 7)
c4 = z4.conjugate()
assert c4 == complex(0, -7)

# Conjugate of zero.
z5 = complex(0, 0)
c5 = z5.conjugate()
assert c5 == complex(0, 0)

# Conjugate preserves magnitude: |z| == |conj(z)|.
z6 = complex(3, 4)
c6 = z6.conjugate()
assert abs(z6) == abs(c6)

# Conjugate of negative components.
z7 = complex(-1, -2)
c7 = z7.conjugate()
assert c7 == complex(-1, 2)

# Conjugate with inf.
z8 = complex(float("inf"), 1)
c8 = z8.conjugate()
assert c8.real == float("inf")
assert c8.imag == -1.0

# Signed zero preservation.
z9 = complex(-0.0, -0.0)
c9 = z9.conjugate()
assert math.copysign(1.0, c9.real) == -1.0
assert math.copysign(1.0, c9.imag) == 1.0
