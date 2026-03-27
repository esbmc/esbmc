def test_dict_value_comparison():
    """Test comparing values during iteration"""
    d: dict[str, int] = {"a": 10, "b": 20, "c": 30}
    count: int = 0
    for v in d.values():
        if v >= 20:
            count = count + 1
    assert count == 1

test_dict_value_comparison()
