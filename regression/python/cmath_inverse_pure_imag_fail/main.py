import cmath

# Negative variant: cmath.acos(0.5j).real is pi/2, not 1.0, so this must FAIL.


def acos_real() -> float:
    z = cmath.acos(0.5j)
    return z.real


assert acos_real() == 1.0
