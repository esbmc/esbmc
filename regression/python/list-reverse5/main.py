def test_reverse_twice_is_identity():
    l = [1, 2, 3, 4, 5]
    l.reverse()
    l.reverse()
    assert l == [1, 2, 3, 4, 5]

test_reverse_twice_is_identity()

