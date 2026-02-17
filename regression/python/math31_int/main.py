import math


def test_math_int_functions() -> None:
    assert math.factorial(5) == 120
    assert math.gcd(12, 18) == 6
    assert math.lcm(4, 6) == 12
    assert math.isqrt(10) == 3
    assert math.perm(5, 2) == 20
    assert math.prod([1, 2, 3, 4]) == 24
    assert math.isclose(1.0, 1.0) == True
    assert math.isclose(1.0, 1.1, rel_tol=1e-9, abs_tol=1e-9) == False


test_math_int_functions()
