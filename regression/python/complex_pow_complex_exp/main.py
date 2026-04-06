# Tests complex power with a complex exponent (not just float/int)
# This exercises: promote_to_complex(rhs) -> complex_log -> complex_mul -> complex_exp

# (1+0j)**(0+1j) = e^(i*ln(1)) = e^0 = 1+0j
w = complex(1, 0) ** complex(0, 1)
assert w.real > 0.99 and w.real < 1.01
assert w.imag > -0.01 and w.imag < 0.01
