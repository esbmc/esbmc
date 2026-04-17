# Small exponent (within budget) uses inline binary exponentiation.
# z**2 = (1+2j)**2 = 1 + 4j + 4j^2 = 1 + 4j - 4 = -3 + 4j
z = complex(1, 2)
w = z ** 2
assert abs(w.real - (-3.0)) < 1e-10
assert abs(w.imag - 4.0) < 1e-10

# z**3 = z**2 * z = (-3+4j)*(1+2j) = -3 -6j +4j +8j^2 = -11 -2j
w3 = z ** 3
assert abs(w3.real - (-11.0)) < 1e-10
assert abs(w3.imag - (-2.0)) < 1e-10
