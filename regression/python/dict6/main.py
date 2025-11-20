def test_string_operations():
    """Should PASS: String operations with dict values"""
    person: dict = {'first': 'John', 'last': 'Doe'}
    
    first_initial: str = person['first']
    assert len(first_initial) > 0  # Verify non-empty

test_string_operations()
