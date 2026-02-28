def saturating_mul(a: int, b: int) -> int:
    return a * b if a > 0 and b > 0 else 0


def saturating_mul_same_sign(a: int, b: int) -> int:
    return a * b if (a >= 0 and b >= 0) or (a < 0 and b < 0) else 0


def test_saturating_mul():
    assert saturating_mul(3, 4) == 12
    assert saturating_mul(2, 5) == 10
    assert saturating_mul(1, 1) == 1
    assert saturating_mul(7, 8) == 56
    assert saturating_mul(0, 5) == 0  # 0 is not > 0, so condition False
    assert saturating_mul(5, 0) == 0  # 0 is not > 0, so condition False
    assert saturating_mul(0, 0) == 0  # both 0, condition False
    assert saturating_mul(0, -5) == 0  # 0 is not > 0, condition False
    assert saturating_mul(-5, 0) == 0  # 0 is not > 0, condition False
    assert saturating_mul(1, 5) == 5  # 1 > 0 and 5 > 0, condition True
    assert saturating_mul(5, 1) == 5  # 1 > 0 and 5 > 0, condition True
    assert saturating_mul(1, 1) == 1  # both > 0, condition True
    assert saturating_mul(-1, 5) == 0  # -1 is not > 0, condition False
    assert saturating_mul(5, -1) == 0  # -1 is not > 0, condition False
    assert saturating_mul(-3, -4) == 0  # both negative, condition False
    assert saturating_mul(-1, -1) == 0  # both negative, condition False
    assert saturating_mul(10, -2) == 0  # -2 is not > 0, condition False
    assert saturating_mul(-2, 10) == 0  # -2 is not > 0, condition False
    assert saturating_mul(-5, 3) == 0  # -5 is not > 0, condition False
    assert saturating_mul(3, -5) == 0  # -5 is not > 0, condition False
    assert saturating_mul(1000, 2000) == 2000000
    assert saturating_mul(999999, 2) == 1999998
    assert saturating_mul(12345, 6789) == 12345 * 6789


def test_saturating_mul_same_sign():
    assert saturating_mul_same_sign(3, 4) == 12
    assert saturating_mul_same_sign(1, 1) == 1
    assert saturating_mul_same_sign(0, 5) == 0  # 0 * 5 = 0, both >= 0
    assert saturating_mul_same_sign(5, 0) == 0  # 5 * 0 = 0, both >= 0
    assert saturating_mul_same_sign(0, 0) == 0  # 0 * 0 = 0, both >= 0
    assert saturating_mul_same_sign(-3, -4) == 12  # (-3) * (-4) = 12
    assert saturating_mul_same_sign(-1, -1) == 1  # (-1) * (-1) = 1
    assert saturating_mul_same_sign(-5, -2) == 10  # (-5) * (-2) = 10
    assert saturating_mul_same_sign(3, -4) == 0  # Different signs
    assert saturating_mul_same_sign(-3, 4) == 0  # Different signs
    assert saturating_mul_same_sign(1, -1) == 0  # Different signs
    assert saturating_mul_same_sign(-1, 1) == 0  # Different signs
    assert saturating_mul_same_sign(0, -5) == 0  # 0 >= 0, -5 < 0, different signs
    assert saturating_mul_same_sign(-5, 0) == 0  # -5 < 0, 0 >= 0, different signs


def test_ternary_operator_edge_cases():
    assert saturating_mul(1, 2) == 2  # True condition
    assert saturating_mul(0, 999999) == 0  # False condition (first part of AND)
    assert saturating_mul(999999, 0) == 0  # False condition (second part of AND)

    # Test that the condition is evaluated correctly for edge values
    tiny_positive = 1e-10  # This would be 0 when converted to int
    assert saturating_mul(int(tiny_positive), 5) == 0  # int(1e-10) = 0, so condition False

    assert saturating_mul(1, 2) == 2  # 1 > 0 and 2 > 0 = True
    assert saturating_mul(1, 0) == 0  # 1 > 0 and 0 > 0 = False
    assert saturating_mul(0, 1) == 0  # 0 > 0 and 1 > 0 = False


test_saturating_mul()
test_saturating_mul_same_sign()
test_ternary_operator_edge_cases()
