import math


def test_math_fail() -> None:
    math.factorial(-1)
    math.isqrt(-1)
    math.acosh(0.5)
    math.atanh(2.0)
    math.log1p(-1.0)
    assert False


test_math_fail()
