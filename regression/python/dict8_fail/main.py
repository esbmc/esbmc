def test_conditional():
    settings: dict = {'mode': 'production', 'debug': False}
    
    mode: str = settings['mode']
    if mode == 'production':
        assert settings['debug'] == True
    else:
        assert False  # Should not reach here

test_conditional()
