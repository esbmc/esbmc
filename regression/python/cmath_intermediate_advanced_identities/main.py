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
atan_r = cmath.atan(complex(x, 0.0))
asinh_r = cmath.asinh(complex(x, 0.0))
atanh_r = cmath.atanh(complex(x, 0.0))
acosh_r = cmath.acosh(complex(2.0, 0.0))

sin_asin = cmath.sin(asin_r)
cos_acos = cmath.cos(acos_r)
tan_atan = cmath.tan(atan_r)
sinh_asinh = cmath.sinh(asinh_r)
tanh_atanh = cmath.tanh(atanh_r)
cosh_acosh = cmath.cosh(acosh_r)

assert approx(sin_asin.real, x, 1e-4)
assert approx(sin_asin.imag, 0.0, 1e-4)
assert approx(cos_acos.real, x, 1e-4)
assert approx(cos_acos.imag, 0.0, 1e-4)
assert approx(tan_atan.real, x, 1e-4)
assert approx(tan_atan.imag, 0.0, 1e-4)
assert approx(sinh_asinh.real, x, 1e-4)
assert approx(sinh_asinh.imag, 0.0, 1e-4)
assert approx(tanh_atanh.real, x, 1e-4)
assert approx(tanh_atanh.imag, 0.0, 1e-4)
assert approx(cosh_acosh.real, 2.0, 1e-4)
assert approx(cosh_acosh.imag, 0.0, 1e-4)

lhs_asinh = cmath.asinh(complex(x, 0.0))
rhs_asinh = cmath.log(complex(x + math.sqrt(x * x + 1.0), 0.0))
assert approx(lhs_asinh.real, rhs_asinh.real, 1e-4)
assert approx(lhs_asinh.imag, rhs_asinh.imag, 1e-4)

lhs_atanh = cmath.atanh(complex(x, 0.0))
rhs_atanh = complex(0.5 * math.log((1.0 + x) / (1.0 - x)), 0.0)
assert approx(lhs_atanh.real, rhs_atanh.real, 1e-4)
assert approx(lhs_atanh.imag, rhs_atanh.imag, 1e-4)

x_id = 0.4
sin_id = cmath.sin(complex(x_id, 0.0))
cos_id = cmath.cos(complex(x_id, 0.0))
sinh_id = cmath.sinh(complex(x_id, 0.0))
cosh_id = cmath.cosh(complex(x_id, 0.0))

sin2_plus_cos2 = sin_id * sin_id + cos_id * cos_id
cosh2_minus_sinh2 = cosh_id * cosh_id - sinh_id * sinh_id
assert approx(sin2_plus_cos2.real, 1.0, 1e-4)
assert approx(sin2_plus_cos2.imag, 0.0, 1e-4)
assert approx(cosh2_minus_sinh2.real, 1.0, 1e-4)
assert approx(cosh2_minus_sinh2.imag, 0.0, 1e-4)
