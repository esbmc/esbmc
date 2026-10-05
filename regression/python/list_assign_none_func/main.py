def test():
    m: list[int] = [1]
    m[0] = None
    assert m[0] is None

test()
