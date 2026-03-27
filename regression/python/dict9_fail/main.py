def test_type_mismatch():
    """Should FAIL: Type mismatch in assertion"""
    data: dict = {'value': 42}
    
    # This should fail verification because types don't match
    assert data['value'] == 'string'  # int != string

test_type_mismatch()
