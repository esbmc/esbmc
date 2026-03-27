def test_integer_values():
    scores: dict = {'alice': 95, 'bob': 87, 'charlie': 92}

    assert scores['alice'] == 94
    assert scores['bob'] < 80
    assert scores['charlie'] < scores['bob']


test_integer_values()
