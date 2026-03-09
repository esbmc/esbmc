import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-5) -> bool:
    return absf(a - b) <= tol


# Pure imaginary inputs: known identities
s_im = cmath.sin(complex(0.0, 1.0))
assert approx(s_im.real, 0.0, 1e-5)
assert s_im.imag > 1.1 and s_im.imag < 1.2

c_im = cmath.cos(complex(0.0, 1.0))
assert c_im.real > 1.5 and c_im.real < 1.6
assert approx(c_im.imag, 0.0, 1e-5)

sh_im = cmath.sinh(complex(0.0, 1.0))
assert approx(sh_im.real, 0.0, 1e-5)
assert sh_im.imag > 0.8 and sh_im.imag < 0.9

ch_im = cmath.cosh(complex(0.0, 1.0))
assert ch_im.real > 0.5 and ch_im.real < 0.6
assert approx(ch_im.imag, 0.0, 1e-5)

# Imaginary-axis identities with real hyperbolic/trigonometric counterparts
y = 0.5
iy = complex(0.0, y)
sin_iy = cmath.sin(iy)
cos_iy = cmath.cos(iy)
tan_iy = cmath.tan(iy)

assert approx(sin_iy.real, 0.0, 1e-5)
assert approx(sin_iy.imag, math.sinh(y), 1e-5)
assert approx(cos_iy.real, math.cosh(y), 1e-5)
assert approx(cos_iy.imag, 0.0, 1e-5)
assert approx(tan_iy.real, 0.0, 1e-5)
assert approx(tan_iy.imag, math.tanh(y), 1e-5)

# Higher-magnitude point: smoke + finite checks
z = complex(2.0, 3.0)
assert cmath.isfinite(cmath.sin(z))
assert cmath.isfinite(cmath.cos(z))
assert cmath.isfinite(cmath.tan(z))
assert cmath.isfinite(cmath.sinh(z))
assert cmath.isfinite(cmath.cosh(z))
assert cmath.isfinite(cmath.tanh(z))

# Parity / symmetry checks
z_par = complex(0.7, -0.4)
z_neg = complex(-0.7, 0.4)

s_par = cmath.sin(z_par)
s_neg = cmath.sin(z_neg)
assert approx(s_neg.real, -s_par.real, 1e-5)
assert approx(s_neg.imag, -s_par.imag, 1e-5)

c_par = cmath.cos(z_par)
c_neg = cmath.cos(z_neg)
assert approx(c_neg.real, c_par.real, 1e-5)
assert approx(c_neg.imag, c_par.imag, 1e-5)

sh_par = cmath.sinh(z_par)
sh_neg = cmath.sinh(z_neg)
assert approx(sh_neg.real, -sh_par.real, 1e-5)
assert approx(sh_neg.imag, -sh_par.imag, 1e-5)

ch_par = cmath.cosh(z_par)
ch_neg = cmath.cosh(z_neg)
assert approx(ch_neg.real, ch_par.real, 1e-5)
assert approx(ch_neg.imag, ch_par.imag, 1e-5)

t_par = cmath.tan(z_par)
t_neg = cmath.tan(z_neg)
assert approx(t_neg.real, -t_par.real, 1e-5)
assert approx(t_neg.imag, -t_par.imag, 1e-5)

th_par = cmath.tanh(z_par)
th_neg = cmath.tanh(z_neg)
assert approx(th_neg.real, -th_par.real, 1e-5)
assert approx(th_neg.imag, -th_par.imag, 1e-5)

# Conjugation symmetry: f(conj(z)) == conj(f(z)) for direct trig/hyper functions
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

sinh_zc = cmath.sinh(zc)
sinh_zc_conj = cmath.sinh(zc_conj)
assert approx(sinh_zc_conj.real, sinh_zc.real, 1e-5)
assert approx(sinh_zc_conj.imag, -sinh_zc.imag, 1e-5)

cosh_zc = cmath.cosh(zc)
cosh_zc_conj = cmath.cosh(zc_conj)
assert approx(cosh_zc_conj.real, cosh_zc.real, 1e-5)
assert approx(cosh_zc_conj.imag, -cosh_zc.imag, 1e-5)

tan_zc = cmath.tan(zc)
tan_zc_conj = cmath.tan(zc_conj)
assert approx(tan_zc_conj.real, tan_zc.real, 1e-5)
assert approx(tan_zc_conj.imag, -tan_zc.imag, 1e-5)

tanh_zc = cmath.tanh(zc)
tanh_zc_conj = cmath.tanh(zc_conj)
assert approx(tanh_zc_conj.real, tanh_zc.real, 1e-5)
assert approx(tanh_zc_conj.imag, -tanh_zc.imag, 1e-5)

# Periodicity for direct trig functions on complex arguments: f(z + 2*pi) = f(z)
two_pi = 2.0 * math.pi
z_per = complex(0.4, 0.2)
s0 = cmath.sin(z_per)
s1 = cmath.sin(z_per + complex(two_pi, 0.0))
c0 = cmath.cos(z_per)
c1 = cmath.cos(z_per + complex(two_pi, 0.0))
t0 = cmath.tan(z_per)
t1 = cmath.tan(z_per + complex(two_pi, 0.0))
assert approx(s0.real, s1.real, 1e-5)
assert approx(s0.imag, s1.imag, 1e-5)
assert approx(c0.real, c1.real, 1e-5)
assert approx(c0.imag, c1.imag, 1e-5)
assert approx(t0.real, t1.real, 1e-5)
assert approx(t0.imag, t1.imag, 1e-5)

# Real-axis consistency with math for direct functions
x = 0.3
sin_real = cmath.sin(complex(x, 0.0))
cos_real = cmath.cos(complex(x, 0.0))
sinh_real = cmath.sinh(complex(x, 0.0))
cosh_real = cmath.cosh(complex(x, 0.0))

assert approx(sin_real.real, math.sin(x), 1e-5)
assert approx(sin_real.imag, 0.0, 1e-5)
assert approx(cos_real.real, math.cos(x), 1e-5)
assert approx(cos_real.imag, 0.0, 1e-5)
assert approx(sinh_real.real, math.sinh(x), 1e-5)
assert approx(sinh_real.imag, 0.0, 1e-5)
assert approx(cosh_real.real, math.cosh(x), 1e-5)
assert approx(cosh_real.imag, 0.0, 1e-5)

# atanh near-boundary on real axis (stable, no exception path)
atanh_hi = cmath.atanh(complex(0.9999, 0.0))
atanh_lo = cmath.atanh(complex(-0.9999, 0.0))
assert atanh_hi.real > 4.0
assert approx(atanh_hi.imag, 0.0, 1e-5)
assert atanh_lo.real < -4.0
assert approx(atanh_lo.imag, 0.0, 1e-5)

# real-axis inverse boundaries
asin_pos1 = cmath.asin(complex(1.0, 0.0))
asin_neg1 = cmath.asin(complex(-1.0, 0.0))
acos_pos1 = cmath.acos(complex(1.0, 0.0))
acos_neg1 = cmath.acos(complex(-1.0, 0.0))
atan_pos1 = cmath.atan(complex(1.0, 0.0))
atan_neg1 = cmath.atan(complex(-1.0, 0.0))

assert approx(asin_pos1.real, math.pi / 2.0, 1e-5)
assert approx(asin_pos1.imag, 0.0, 1e-5)
assert approx(asin_neg1.real, -math.pi / 2.0, 1e-5)
assert approx(asin_neg1.imag, 0.0, 1e-5)
assert approx(acos_pos1.real, 0.0, 1e-5)
assert approx(acos_pos1.imag, 0.0, 1e-5)
assert approx(acos_neg1.real, math.pi, 1e-5)
assert approx(acos_neg1.imag, 0.0, 1e-5)
assert approx(atan_pos1.real, math.pi / 4.0, 1e-5)
assert approx(atan_pos1.imag, 0.0, 1e-5)
assert approx(atan_neg1.real, -math.pi / 4.0, 1e-5)
assert approx(atan_neg1.imag, 0.0, 1e-5)
