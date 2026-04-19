import math


def test_math_dist_fail() -> None:
    math.dist([0.0, 0.0], [1.0])
    assert False


test_math_dist_fail()
