def test_basic_dict():
    """Should PASS: Basic dictionary operations"""
    dict = {'name': 'Bob', 'ref': 'Python', 'sys': 'Win'}
    assert dict['ref'] == 'Python'
    assert dict['name'] == 'Bob'
    assert dict['sys'] == 'Win'


test_basic_dict()
