def test_string_operations():
    person: dict = {'first': 'John', 'last': 'Doe'}
    
    first_initial: str = person['first']
    assert len(first_initial) < 0

test_string_operations()
