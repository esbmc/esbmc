def test_delete_string_key():
    d = {"x": 1, "y": 2, "z": 3}
    del d["y"]
    assert "x" in d
    assert "y" not in d
    assert "z" in d


test_delete_string_key()
