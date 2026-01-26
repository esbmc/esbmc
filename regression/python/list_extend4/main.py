def test_extend_aliasing():
    a = [1]
    b = [2]
    a.extend(b)
    b.append(3)
    assert a == [1, 2]   # a should not change after b is modified

test_extend_aliasing()
