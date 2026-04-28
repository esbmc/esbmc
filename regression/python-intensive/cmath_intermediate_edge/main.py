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
