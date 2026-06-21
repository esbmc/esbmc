import cmath

# Regression test for bare pure-imaginary literal passed directly as a
# by-value complex argument.  Before the fix, `0.5j` was lowered to a
# string literal "0.5j" (because the JSON `value` field holds the Python
# str representation) and then wrapped in address_of(), producing a
# "got pointer, expected struct" SIGABRT on entry to the callee.

# Plain user function: the general argument-binding path
def get_imag(z: complex) -> float:
    return z.imag

def get_real(z: complex) -> float:
    return z.real

# Bare imaginary literals passed directly as call arguments
assert get_imag(0.5j) == 0.5
assert get_real(0.5j) == 0.0
assert get_imag(1j) == 1.0
assert get_real(1j) == 0.0

# cmath functions with bare imaginary literals (all were crashing before)
r1 = cmath.exp(0.5j)
r2 = cmath.sqrt(0.5j)
r3 = cmath.sin(0.5j)
r4 = cmath.phase(0.5j)
