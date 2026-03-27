def test_multiple_assertions():
    data: dict = {'value': 10}
    
    x: int = data['value']
    assert x != 10
    assert x < 5
    assert x > 15
    assert x == 0

test_multiple_assertions()
