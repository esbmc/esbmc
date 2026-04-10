import math


def test_math29_success() -> None:
    # Trigonometric
    assert math.tan(0.0) == 0.0
    assert math.asin(0.0) == 0.0
    assert math.acos(1.0) == 0.0
    assert math.atan(0.0) == 0.0
    assert math.atan2(0.0, 1.0) == 0.0

    # Hyperbolic
    assert math.sinh(0.0) == 0.0
    assert math.cosh(0.0) == 1.0
    assert math.tanh(0.0) == 0.0

    # Power and logarithmic
    assert math.pow(2.0, 3.0) == 8.0
    assert math.isfinite(math.log2(8.0))
    assert math.isfinite(math.log10(100.0))

    # Rounding and absolute value
    assert math.fabs(-2.5) == 2.5
    assert math.trunc(3.7) == 3
    frac, integer = math.modf(3.25)
    assert math.isfinite(frac)
    assert math.isfinite(integer)

    # Other common functions
    assert math.fmod(7.0, 3.0) == 1.0
    assert math.copysign(1.5, -2.0) == -1.5
    assert math.isfinite(1.0) == True
    assert math.isfinite(math.degrees(math.pi))
    assert math.isfinite(math.radians(180.0))


test_math29_success()
