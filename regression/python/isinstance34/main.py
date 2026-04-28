def test_isinstance_none():
    assert not isinstance(None, int)
    assert not isinstance(None, float)
    assert not isinstance(None, bool)
    assert not isinstance(None, str)

test_isinstance_none()