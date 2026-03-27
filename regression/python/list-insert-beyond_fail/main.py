def test_insert():
    l = [1, 3]
    l.insert(100, 4)
    assert l[-1] == 3


test_insert()
