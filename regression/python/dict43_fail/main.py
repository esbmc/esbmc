def test_nested_dict_access():
    d = {"user": {"name": "Alice", "age": 30}}
    e = {"user": {"name": "Alice", "age": 30}}

    assert d is e


test_nested_dict_access()
