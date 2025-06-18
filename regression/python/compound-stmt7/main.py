def test_all_compound_ops():
    a = 4
    a %= 3     # 1
    assert a == 1
    a **= 3    # 1
    assert a == 1

test_all_compound_ops()

