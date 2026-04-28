import math


def test_math_helpers() -> None:
    assert math.gcd(-6, 9) == 3
    assert math.lcm(-3, 4) == 12
    assert math.prod([1, 2, 3], start=2) == 12
    assert math.isclose(1.0, 1.0000001, rel_tol=1e-5, abs_tol=0.0)


test_math_helpers()
