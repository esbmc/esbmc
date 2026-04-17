import math


def test_math_exp_functions() -> None:
    assert math.expm1(0.0) == 0.0
    assert math.log1p(0.0) == 0.0
    assert math.isclose(math.exp2(3.0), 8.0, rel_tol=1e-6, abs_tol=1e-6)


test_math_exp_functions()
