def test_rounding() -> None:
    assert round(3.6) == 4
    assert round(3.14159, 2) == 3.14
    assert round(-3.6) == -4
    assert round(-2.5) == -2
    assert round(0.5) == 0
    assert round(1.5) == 2
    assert round(2.5) == 2


test_rounding()
