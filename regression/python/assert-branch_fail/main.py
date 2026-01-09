def test_branch(x: int):
    if x > 10:
        assert x < 20
    return x

test_branch(25)
