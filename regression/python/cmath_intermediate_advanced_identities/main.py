import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol


x = 0.25
asin_r = cmath.asin(complex(x, 0.0))
acos_r = cmath.acos(complex(x, 0.0))

sin_asin = cmath.sin(asin_r)
cos_acos = cmath.cos(acos_r)

assert approx(sin_asin.real, x, 1e-4)
assert approx(sin_asin.imag, 0.0, 1e-4)
assert approx(cos_acos.real, x, 1e-4)
assert approx(cos_acos.imag, 0.0, 1e-4)

x_id = 0.4
sin_id = cmath.sin(complex(x_id, 0.0))
cos_id = cmath.cos(complex(x_id, 0.0))
sin2_plus_cos2 = sin_id * sin_id + cos_id * cos_id
assert approx(sin2_plus_cos2.real, 1.0, 1e-4)
assert approx(sin2_plus_cos2.imag, 0.0, 1e-4)
