def test_mixed_types():
    """Should PASS: Dictionary with different value types"""
    config: dict = {'debug': True, 'port': 8080, 'host': 'localhost', 'timeout': 30.5}

    assert config['debug'] == True
    assert config['port'] == 8080
    assert config['host'] == 'localhost'
    assert config['timeout'] > 30.0


test_mixed_types()
