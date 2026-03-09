import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol


z = complex(0.1, -0.2)

# direct trig/hyper values stay finite
s = cmath.sin(z)
c = cmath.cos(z)
t = cmath.tan(z)
assert cmath.isfinite(s)
assert cmath.isfinite(c)
assert cmath.isfinite(t)

sh = cmath.sinh(z)
ch = cmath.cosh(z)
th = cmath.tanh(z)
assert cmath.isfinite(sh)
assert cmath.isfinite(ch)
assert cmath.isfinite(th)

# additional edge points
z0 = complex(0.0, 0.0)
assert cmath.asin(z0) == complex(0.0, 0.0)
assert cmath.atan(z0) == complex(0.0, 0.0)
assert cmath.asinh(z0) == complex(0.0, 0.0)
assert cmath.atanh(z0) == complex(0.0, 0.0)

assert cmath.sin(z0) == complex(0.0, 0.0)
assert cmath.tan(z0) == complex(0.0, 0.0)
assert cmath.sinh(z0) == complex(0.0, 0.0)
assert cmath.tanh(z0) == complex(0.0, 0.0)

c0 = cmath.cos(z0)
assert approx(c0.real, 1.0, 1e-6)
assert approx(c0.imag, 0.0, 1e-6)

ch0 = cmath.cosh(z0)
assert approx(ch0.real, 1.0, 1e-6)
assert approx(ch0.imag, 0.0, 1e-6)

# inverse functions on real-axis non-trivial points
x = 0.25
asin_r = cmath.asin(complex(x, 0.0))
acos_r = cmath.acos(complex(x, 0.0))
atan_r = cmath.atan(complex(x, 0.0))
asinh_r = cmath.asinh(complex(x, 0.0))
atanh_r = cmath.atanh(complex(x, 0.0))
acosh_r = cmath.acosh(complex(2.0, 0.0))

assert approx(asin_r.real, math.asin(x), 1e-4)
assert approx(asin_r.imag, 0.0, 1e-4)
assert approx(acos_r.real, math.acos(x), 1e-4)
assert approx(acos_r.imag, 0.0, 1e-4)
assert approx(atan_r.real, math.atan(x), 1e-4)
assert approx(atan_r.imag, 0.0, 1e-4)
assert approx(asinh_r.real, math.asinh(x), 1e-4)
assert approx(asinh_r.imag, 0.0, 1e-4)
assert approx(atanh_r.real, math.atanh(x), 1e-4)
assert approx(atanh_r.imag, 0.0, 1e-4)
assert approx(acosh_r.real, math.acosh(2.0), 1e-4)
assert approx(acosh_r.imag, 0.0, 1e-4)

# real-axis inverse identities and symmetry
asin_neg_r = cmath.asin(complex(-x, 0.0))
acos_neg_r = cmath.acos(complex(-x, 0.0))
atan_neg_r = cmath.atan(complex(-x, 0.0))
asinh_neg_r = cmath.asinh(complex(-x, 0.0))
atanh_neg_r = cmath.atanh(complex(-x, 0.0))

assert approx(asin_neg_r.real, -asin_r.real, 1e-4)
assert approx(asin_neg_r.imag, -asin_r.imag, 1e-4)
assert approx(acos_neg_r.real, math.pi - acos_r.real, 1e-4)
assert approx(acos_neg_r.imag, -acos_r.imag, 1e-4)
assert approx(atan_neg_r.real, -atan_r.real, 1e-4)
assert approx(atan_neg_r.imag, -atan_r.imag, 1e-4)
assert approx(asinh_neg_r.real, -asinh_r.real, 1e-4)
assert approx(asinh_neg_r.imag, -asinh_r.imag, 1e-4)
assert approx(atanh_neg_r.real, -atanh_r.real, 1e-4)
assert approx(atanh_neg_r.imag, -atanh_r.imag, 1e-4)

sum_asin_acos = asin_r + acos_r
assert approx(sum_asin_acos.real, math.pi / 2.0, 1e-4)
assert approx(sum_asin_acos.imag, 0.0, 1e-4)

# complementary-angle identities on real axis
sum_acos = acos_r + acos_neg_r
assert approx(sum_acos.real, math.pi, 1e-4)
assert approx(sum_acos.imag, 0.0, 1e-4)

sum_asin = asin_r + asin_neg_r
assert approx(sum_asin.real, 0.0, 1e-4)
assert approx(sum_asin.imag, 0.0, 1e-4)

# composition identities on stable real-axis points
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

# analytical identities on real-axis points (stable)
lhs_asinh = cmath.asinh(complex(x, 0.0))
rhs_asinh = cmath.log(complex(x + math.sqrt(x * x + 1.0), 0.0))
assert approx(lhs_asinh.real, rhs_asinh.real, 1e-4)
assert approx(lhs_asinh.imag, rhs_asinh.imag, 1e-4)

lhs_atanh = cmath.atanh(complex(x, 0.0))
rhs_atanh = complex(0.5 * math.log((1.0 + x) / (1.0 - x)), 0.0)
assert approx(lhs_atanh.real, rhs_atanh.real, 1e-4)
assert approx(lhs_atanh.imag, rhs_atanh.imag, 1e-4)

# real-axis Pythagorean/Lorentz identities
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
