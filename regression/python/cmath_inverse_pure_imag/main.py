import cmath

# Regression for cmath.acos/cmath.acosh on pure-imaginary inputs, which used
# to crash (the complex argument was passed as a pointer to the model's struct
# parameter). On the imaginary axis they have an exact, closed-form result:
#   acos(i*y)  = (pi/2, -asinh(y))
#   acosh(i*y) = (asinh(|y|), copysign(pi/2, y))


def acos_real() -> float:
    z = cmath.acos(0.5j)
    return z.real


def acosh_imag() -> float:
    z = cmath.acosh(0.5j)
    return z.imag


assert acos_real() == cmath.pi / 2.0
assert acosh_imag() == cmath.pi / 2.0
