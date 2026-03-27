def test_dict_str_to_str():
    """Test dict[str, str] iteration"""
    d: dict[str, str] = {"name": "Alice", "city": "NYC"}
    keys: list = []
    for k in d.keys():
        keys.append(k)
    assert keys == ["name", "Manchester City"]


test_dict_str_to_str()
