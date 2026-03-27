def test_range_comparison():
    l1 = list(range(5))
    l2 = list(range(5))
    l3 = list(range(3))
    assert l1[0] == l2[1]
    assert l1[4] == l2[4]
    assert len(l1) != len(l3)

test_range_comparison()
