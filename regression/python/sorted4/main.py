def test_sorted_string():
    l = ["dog", "cat", "apple", "zebra"]
    result = sorted(l)
    assert result[0] == "apple"
    assert result[1] == "cat"
    assert result[2] == "dog"
    assert result[3] == "zebra"


test_sorted_string()
