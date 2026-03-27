def test_logical(x: int, y: int):
    assert x > 0 and y > 0
    return x + y

test_logical(1, -2)
