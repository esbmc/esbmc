def test_chained_comparisons():
    x = 0
    assert 0 <= x <= 2 < 3, "Failed at x = 0"

    x = 1
    assert 0 <= x <= 2 < 3, "Failed at x = 1"

    x = 2
    assert 0 <= x <= 2 < 3, "Failed at x = 2"

    x = -1
    assert 0 <= x <= 2 < 3, "Failed at x = -1 (should fail)"

    x = 3
    assert 0 <= x <= 2 < 3, "Failed at x = 3 (should fail)"

    x = 2.5
    assert 0 <= x <= 2 < 3, "Failed at x = 2.5 (should fail)"

    x = 2
    assert 0 <= x < 2 < 3 == False, "Failed: x = 2 should not satisfy x < 2"

test_chained_comparisons()
