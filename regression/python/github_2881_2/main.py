def test_not_in_behavior() -> None:
    assert 'a' not in 'xyz'
    assert 'x' not in ['y', 'z']
    assert not ('a' not in 'abc')


test_not_in_behavior()
