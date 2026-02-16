import math


def test_math_nondet() -> None:
    x = nondet_float()
    y = nondet_float()
    z = nondet_float()

    # Trigonometric
    __ESBMC_assume(x >= -0.5 and x <= 0.5)
    t = math.tan(x)
    assert math.isfinite(t)
    assert t >= -2.0 and t <= 2.0

    __ESBMC_assume(y >= -1.0 and y <= 1.0)
    a = math.asin(y)
    assert math.isfinite(a)
    assert a >= -2.0 and a <= 2.0

    __ESBMC_assume(z >= -1.0 and z <= 1.0)
    c = math.acos(z)
    assert math.isfinite(c)
    assert c >= 0.0 and c <= 4.0

    # atan/atan2 with bounded inputs
    __ESBMC_assume(x >= -1.0 and x <= 1.0)
    at = math.atan(x)
    assert math.isfinite(at)
    assert at >= -2.0 and at <= 2.0

    __ESBMC_assume(y >= -1.0 and y <= 1.0)
    __ESBMC_assume(z >= -1.0 and z <= 1.0)
    at2 = math.atan2(y, z)
    assert math.isfinite(at2)
    assert at2 >= -4.0 and at2 <= 4.0

    # Hyperbolic
    __ESBMC_assume(x >= -1.0 and x <= 1.0)
    sh = math.sinh(x)
    ch = math.cosh(x)
    th = math.tanh(x)
    assert math.isfinite(sh)
    assert math.isfinite(ch)
    assert math.isfinite(th)
    assert ch >= 1.0
    assert th >= -1.0 and th <= 1.0

    # Power and logarithmic
    __ESBMC_assume(x >= 0.5 and x <= 2.0)
    __ESBMC_assume(y >= 0.0 and y <= 3.0)
    p = math.pow(x, y)
    assert math.isfinite(p)
    assert p >= 0.125 and p <= 8.0

    __ESBMC_assume(z >= 0.5 and z <= 4.0)
    l2 = math.log2(z)
    l10 = math.log10(z)
    assert math.isfinite(l2)
    assert math.isfinite(l10)
    assert l2 >= -2.0 and l2 <= 3.0
    assert l10 >= -2.0 and l10 <= 2.0

    # Rounding and absolute value
    __ESBMC_assume(x >= -10.0 and x <= 10.0)
    assert math.fabs(x) >= 0.0
    __ESBMC_assume(y >= -3.9 and y <= 3.9)
    tr = math.trunc(y)
    assert tr >= -3 and tr <= 3

    # modf: just check finiteness
    frac, integer = math.modf(3.25)
    assert math.isfinite(frac)
    assert math.isfinite(integer)

    # Other functions
    __ESBMC_assume(x >= -5.0 and x <= 5.0)
    __ESBMC_assume(y >= 1.0 and y <= 5.0)
    fm = math.fmod(x, y)
    assert math.isfinite(fm)
    assert fm >= -5.0 and fm <= 5.0

    __ESBMC_assume(x >= 0.5 and x <= 2.0)
    __ESBMC_assume(y >= -2.0 and y <= -0.5)
    cs = math.copysign(x, y)
    assert cs <= 0.0

    # Degrees/radians
    __ESBMC_assume(z >= -3.14 and z <= 3.14)
    deg = math.degrees(z)
    assert math.isfinite(deg)
    assert deg >= -200.0 and deg <= 200.0

    __ESBMC_assume(y >= -180.0 and y <= 180.0)
    rad = math.radians(y)
    assert math.isfinite(rad)
    assert rad >= -4.0 and rad <= 4.0


test_math_nondet()
