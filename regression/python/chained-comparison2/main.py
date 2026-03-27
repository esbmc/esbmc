def test_chained_comparisons():
    x = 0
    assert 0 <= x <= 2 < 3, "Failed at x = 0"

    x = 1
    assert 0 <= x <= 2 < 3, "Failed at x = 1"

    x = 2
    assert 0 <= x <= 2 < 3, "Failed at x = 2"

test_chained_comparisons()
print("All assertions passed.")
