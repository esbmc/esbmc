def test_range_two_args():
    l = list(range(2, 7))
    assert len(l) == 5
    assert l[0] == 2
    assert l[1] == 3
    assert l[2] == 4
    assert l[3] == 5
    assert l[4] == 6


test_range_two_args()
