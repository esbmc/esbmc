def test_nested(x: int, y: int):
    if x > 0:
        if y > 0:
            assert x + y < 10
    return x + y

test_nested(6, 5)
