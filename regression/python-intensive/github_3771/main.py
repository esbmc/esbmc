import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol

# Imaginary-axis inverse identities on non-real path
y = 0.2
iy = complex(0.0, y)

asin_iy = cmath.asin(iy)
assert approx(asin_iy.real, 0.0, 1e-4)
assert approx(asin_iy.imag, math.asinh(y), 1e-4)

atan_iy = cmath.atan(iy)
assert approx(atan_iy.real, 0.0, 1e-4)
assert approx(atan_iy.imag, math.atanh(y), 1e-4)

asinh_iy = cmath.asinh(iy)
assert approx(asinh_iy.real, 0.0, 1e-4)
assert approx(asinh_iy.imag, math.asin(y), 1e-4)

atanh_iy = cmath.atanh(iy)
assert approx(atanh_iy.real, 0.0, 1e-4)
assert approx(atanh_iy.imag, math.atan(y), 1e-4)
