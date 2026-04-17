def test_range_two_args():
    l = list(range(2, 7))
    assert len(l) == 5
    assert l[0] == 3
    assert l[1] == 4
    assert l[2] == 5
    assert l[3] == 6
    assert l[4] == 7

test_range_two_args()
