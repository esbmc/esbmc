import math


def test_math_hyper_inv() -> None:
    assert math.asinh(0.0) == 0.0
    assert math.acosh(1.0) == 0.0
    assert math.atanh(0.0) == 0.0


test_math_hyper_inv()
