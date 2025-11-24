def test_mixed_compound_ops():
    x = 2.0
    y = 3.0
    x *= 3     # 6
    y += 2     # 5.0
    x -= 1     # 5
    assert x == 5
    assert y >= 5.0

test_mixed_compound_ops()
