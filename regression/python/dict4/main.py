def test_integer_values():
    """Should PASS: Dictionary with integer values"""
    scores: dict = {'alice': 95, 'bob': 87, 'charlie': 92}

    assert scores['alice'] == 95
    assert scores['bob'] < 90
    assert scores['charlie'] > scores['bob']


test_integer_values()
