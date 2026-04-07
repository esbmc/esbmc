# Negative exponent within budget (|exp| <= 16): uses inline expansion + division.
z = complex(0, 1)

# i^(-1) = -i
w1 = z ** (-1)
assert abs(w1.real - 0.0) < 1e-10
assert abs(w1.imag - (-1.0)) < 1e-10

# i^(-4) = 1/(i^4) = 1/1 = 1
w4 = z ** (-4)
assert abs(w4.real - 1.0) < 1e-10
assert abs(w4.imag - 0.0) < 1e-10
