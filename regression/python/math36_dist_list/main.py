import math


def test_math_dist_list() -> None:
    assert math.isclose(math.dist([0.0, 0.0], [3.0, 4.0]), 5.0, rel_tol=1e-6, abs_tol=1e-6)


test_math_dist_list()
