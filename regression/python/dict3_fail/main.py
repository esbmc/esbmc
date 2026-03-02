def test_mixed_types():
    config: dict = {'debug': True, 'port': 8080, 'host': 'localhost', 'timeout': 30.5}

    assert config['debug'] == False
    assert config['port'] == 8081
    assert config['host'] == 'llocalhost'
    assert config['timeout'] > 31.0


test_mixed_types()
