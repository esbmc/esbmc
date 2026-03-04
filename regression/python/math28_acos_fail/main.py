import math


def test_acos_fail() -> None:
    x = 2.0
    math.acos(x)
    assert False


test_acos_fail()
