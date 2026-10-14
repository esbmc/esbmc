def test_negative_keys():
    d = {-1: 2}
    assert d[-1] == 2


def test_pop():
    d = {-1: 2, 12: 24}
    assert d.pop(-1) == 2
    assert d.pop(7, 8) == 8
    assert d.pop(12, 9) == 24
    assert len(d) == 0


def test_setdefault():
    a = {}
    a.setdefault(1, []).append(1.0)
    assert a == {1: [1.0]}


def test_all():
    test_negative_keys()
    test_pop()
    test_setdefault()


test_all()
