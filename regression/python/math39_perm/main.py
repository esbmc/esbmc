import math


def test_math_perm() -> None:
    assert math.perm(5, 2) == 20
    assert math.perm(5) == 120


test_math_perm()
