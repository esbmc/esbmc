def test_sorted_string():
    l = ["dog", "cat", "apple", "zebra"]
    result = sorted(l)
    assert result[0] == "apple"
    assert result[1] == "cat"
    assert result[2] == "zebra"
    assert result[3] == "dog"

test_sorted_string()
