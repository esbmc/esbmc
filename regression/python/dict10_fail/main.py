def test_key_error():
    """Should ERROR: Accessing non-existent key"""
    data: dict = {'a': 1, 'b': 2}
    
    # This should cause a KeyError at compile time
    x: int = data['c']  # Key 'c' doesn't exist
    assert x == 1

test_key_error()
