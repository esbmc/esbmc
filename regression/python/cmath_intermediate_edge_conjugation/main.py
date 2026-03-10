import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol


zc = complex(0.6, -0.35)
zc_conj = complex(zc.real, -zc.imag)

sin_zc = cmath.sin(zc)
sin_zc_conj = cmath.sin(zc_conj)
assert approx(sin_zc_conj.real, sin_zc.real, 1e-5)
assert approx(sin_zc_conj.imag, -sin_zc.imag, 1e-5)

cos_zc = cmath.cos(zc)
cos_zc_conj = cmath.cos(zc_conj)
assert approx(cos_zc_conj.real, cos_zc.real, 1e-5)
assert approx(cos_zc_conj.imag, -cos_zc.imag, 1e-5)

tan_zc = cmath.tan(zc)
tan_zc_conj = cmath.tan(zc_conj)
assert approx(tan_zc_conj.real, tan_zc.real, 1e-5)
assert approx(tan_zc_conj.imag, -tan_zc.imag, 1e-5)
