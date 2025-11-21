def test_boolean_logic():
    flags: dict = {'enabled': True, 'verified': False, 'active': True}
    
    assert flags['enabled'] == True
    assert flags['verified'] == False
    
    # Combine with boolean operators
    if flags['enabled'] and flags['active']:
        assert False
    else:
        assert True

test_boolean_logic()
