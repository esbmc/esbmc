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

# cmath function with a bare imaginary literal (was crashing before): the
# crash was at argument binding, before the callee runs, so one cheap call
# (phase -> atan2) is enough to pin the cmath dispatch path without dragging
# in the heavy libm series loops of exp/sqrt/sin.
angle = cmath.phase(0.5j)
assert angle > 0.0
