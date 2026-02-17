import math


def test_math_geom() -> None:
    assert math.hypot(3.0, 4.0) == 5.0
    assert math.dist([0.0, 0.0], [3.0, 4.0]) == 5.0


test_math_geom()
