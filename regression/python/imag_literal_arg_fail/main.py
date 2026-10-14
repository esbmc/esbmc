# Negative test: asserts a wrong value for the imaginary part of 0.5j.
# ESBMC must detect the violated assertion.

def get_imag(z: complex) -> float:
    return z.imag

# 0.5j has imag == 0.5, not 99.0
assert get_imag(0.5j) == 99.0
