def test_conditional():
    """Should PASS: Using dict values in if statements"""
    settings: dict = {'mode': 'production', 'debug': False}
    
    mode: str = settings['mode']
    if mode == 'production':
        assert settings['debug'] == False
    else:
        assert False  # Should not reach here

test_conditional()
