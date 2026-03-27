def test_boolean_logic():
    """Should PASS: Boolean expressions with dict values"""
    flags: dict = {'enabled': True, 'verified': False, 'active': True}

    assert flags['enabled'] == True
    assert flags['verified'] == False

    # Combine with boolean operators
    if flags['enabled'] and flags['active']:
        assert True  # Should reach here
    else:
        assert False  # Should not reach here


test_boolean_logic()
