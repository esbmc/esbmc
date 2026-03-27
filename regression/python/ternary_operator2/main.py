def saturating_sub(a: int, b: int) -> int:
    """
    Computes a - b, saturating at numeric bounds.
    """
    return a - b if a > b else 0


def test_saturating_sub():
    assert saturating_sub(3, 1) == 2
    assert saturating_sub(5, 5) == 0
    assert saturating_sub(0, 0) == 0
    assert saturating_sub(-3, -3) == 0
    assert saturating_sub(4, 5) == 0
    assert saturating_sub(-1, 0) == 0
    assert saturating_sub(5, 4) == 1
    assert saturating_sub(0, -1) == 1
    assert saturating_sub(1, 0) == 1
    assert saturating_sub(0, 5) == 0  # 0 < 5, return 0
    assert saturating_sub(5, 0) == 5  # 5 > 0, return 5-0=5
    assert saturating_sub(0, -5) == 5  # 0 > -5, return 0-(-5)=5
    assert saturating_sub(-5, 0) == 0  # -5 < 0, return 0
    assert saturating_sub(-1, -3) == 2  # -1 > -3, return -1-(-3)=2
    assert saturating_sub(-3, -1) == 0  # -3 < -1, return 0
    assert saturating_sub(-10, -5) == 0  # -10 < -5, return 0
    assert saturating_sub(-5, -10) == 5  # -5 > -10, return -5-(-10)=5
    assert saturating_sub(1000000, 999999) == 1
    assert saturating_sub(999999, 1000000) == 0
    assert saturating_sub(-999999, -1000000) == 1
    assert saturating_sub(-1000000, -999999) == 0
    assert saturating_sub(1000000, -1000000) == 2000000
    assert saturating_sub(-1000000, 1000000) == 0
    print("All test cases passed! ✓")


test_saturating_sub()
