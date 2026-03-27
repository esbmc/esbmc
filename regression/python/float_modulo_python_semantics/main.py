def test_result_sign_matches_positive_divisor():
    # Negative dividend, positive divisor -> positive result
    x: float = -10.0
    y: float = 3.0
    z: float = x % y
    assert z == 2.0  # -10 = 3*(-4) + 2


def test_result_sign_matches_negative_divisor():
    # Positive dividend, negative divisor -> negative result
    x: float = 10.0
    y: float = -3.0
    z: float = x % y
    assert z == -2.0  # 10 = -3*(-4) + (-2)


def test_classic_python_example():
    # The classic example showing Python vs C difference
    x: float = -7.0
    y: float = 3.0
    z: float = x % y
    assert z == 2.0  # Python: -7 % 3 = 2, C fmod: -1


def test_another_sign_example():
    x: float = 17.0
    y: float = -5.0
    z: float = x % y
    assert z == -3.0  # Python: 17 % -5 = -3, C fmod: 2


def test_fractional_negative():
    x: float = -8.5
    y: float = 3.0
    z: float = x % y
    assert z == 0.5  # -8.5 = 3*(-3) + 0.5


test_result_sign_matches_positive_divisor()
test_result_sign_matches_negative_divisor()
test_classic_python_example()
test_another_sign_example()
test_fractional_negative()
