import math


def test_math_constants() -> None:
    assert math.tau == 2.0 * math.pi
    assert math.nan != math.nan


test_math_constants()
