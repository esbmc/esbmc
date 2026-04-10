# Tests for budget limit on complex exponentiation (exponent > 16 uses exp/log path).
import math

# Exponent within inline budget (<=16): exact binary exponentiation.
z1 = complex(1, 1)
w1 = z1 ** 10
# (1+1j)**10 = 32j (via binomial, real=0, imag=32)
# Actually: (1+i)^2=2i, (1+i)^4=-4, (1+i)^8=16, (1+i)^10=16*(2i)=32i
assert abs(w1.real - 0.0) < 1e-6
assert abs(w1.imag - 32.0) < 1e-6

# Exponent at budget boundary (16): still inline.
z2 = complex(1, 0)
w2 = z2 ** 16
assert abs(w2.real - 1.0) < 1e-10
assert abs(w2.imag - 0.0) < 1e-10

# Exponent beyond budget (17): uses exp(n*log(z)) fallback.
z3 = complex(1, 0)
w3 = z3 ** 17
assert abs(w3.real - 1.0) < 1e-6
assert abs(w3.imag - 0.0) < 1e-6

# Large exponent (50): must use exp/log path.
z4 = complex(1, 0)
w4 = z4 ** 50
assert abs(w4.real - 1.0) < 1e-6
assert abs(w4.imag - 0.0) < 1e-6

# Negative large exponent: exp/log path with inversion.
z5 = complex(1, 0)
w5 = z5 ** (-20)
assert abs(w5.real - 1.0) < 1e-6
assert abs(w5.imag - 0.0) < 1e-6
