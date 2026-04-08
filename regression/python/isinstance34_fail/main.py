def test_isinstance_none():
    assert not isinstance(None, int)
    assert not isinstance(None, float)
    assert not isinstance(None, bool)
    assert not isinstance(None, str)
    assert not isinstance(None, object)
    assert not isinstance(None, type(None))
    assert not isinstance(None, (int, type(None)))

test_isinstance_none()