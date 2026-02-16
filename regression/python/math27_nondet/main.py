import math


def test_math_nondet() -> None:
    x = nondet_float()
    y = nondet_float()

    # Keep symbolic tests lightweight to avoid timeouts
    __ESBMC_assume(x >= -10.0 and x <= 10.0)
    __ESBMC_assume(y >= 0.5 and y <= 5.0)

    # Rounding and absolute value
    assert math.fabs(x) >= 0.0
    tr = math.trunc(x)
    assert tr >= -10 and tr <= 10

    # fmod and copysign
    fm = math.fmod(x, y)
    assert fm >= -10.0 and fm <= 10.0
    cs = math.copysign(1.0, x)
    assert math.fabs(cs) == 1.0

    # Degrees/radians with symbolic input
    deg = math.degrees(x)
    rad = math.radians(x)
    assert math.isfinite(deg)
    assert math.isfinite(rad)


test_math_nondet()
