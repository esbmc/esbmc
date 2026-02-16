import math


def test_math29_fail() -> None:
    # Domain errors
    math.asin(2.0)
    math.acos(2.0)
    math.log2(0.0)
    math.log10(0.0)
    assert False


test_math29_fail()
