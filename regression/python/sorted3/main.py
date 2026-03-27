def test_sorted_float():
    l = [3.5, 1.2, 2.8, 0.5]
    result = sorted(l)
    assert result[0] == 0.5
    assert result[1] == 1.2
    assert result[2] == 2.8
    assert result[3] == 3.5


test_sorted_float()
