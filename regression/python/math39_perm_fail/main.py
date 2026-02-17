import math


def test_math_perm_fail() -> None:
    math.perm(5, 2.0)
    assert False


test_math_perm_fail()
