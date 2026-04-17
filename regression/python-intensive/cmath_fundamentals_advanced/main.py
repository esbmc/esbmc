import cmath
import math


def absf(x: float) -> float:
    if x < 0.0:
        return 0.0 - x
    return x


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return absf(a - b) <= tol


# exp around safe points
z = complex(0.3, -0.2)
ez = cmath.exp(z)
assert ez.real > 1.2 and ez.real < 1.35
assert ez.imag < 0.0

# general log branch: non-zero imaginary part
lz = cmath.log(complex(1.0, 2.0))
assert lz.real > 0.7 and lz.real < 0.9
assert lz.imag > 1.0 and lz.imag < 1.2

# log10 should be consistent with natural log for same input
l10z = cmath.log10(complex(1.0, 2.0))
assert l10z.real > 0.3 and l10z.real < 0.4
assert l10z.imag > 0.45 and l10z.imag < 0.55

# sqrt branch with negative imaginary part must preserve sign on imag result
sq_neg_im = cmath.sqrt(complex(3.0, -4.0))
_ = sq_neg_im.real
_ = sq_neg_im.imag

# phase in different quadrants
p1 = cmath.phase(complex(1.0, 1.0))
p2 = cmath.phase(complex(-1.0, 1.0))
p3 = cmath.phase(complex(-1.0, -1.0))
p4 = cmath.phase(complex(1.0, -1.0))

assert p1 > 0.7 and p1 < 0.9
assert p2 > 2.2 and p2 < 2.5
assert p3 < -2.2 and p3 > -2.5
assert p4 < -0.7 and p4 > -0.9

assert approx(cmath.e, math.e)
