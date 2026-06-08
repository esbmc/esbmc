def test_dict_get() -> None:
    assert {"wah": 2}.get("aap", 3) == 3


def test_dict_del() -> None:
    d = {1: 4, 2: 5}
    del d[1]
    assert d == {2: 5}


def test_mixed_numeric_keys() -> None:
    a = {}
    a[1.0] = 1
    assert a[1.0] == 1

    b = {}
    b[1] = 1.0
    assert b[1] == 1.0

    c = {}
    c[4] = 1.0
    assert c[4] == 1.0
    assert 4 in c


def test_negative_keys() -> None:
    d = {-1: 2}
    assert d[-1] == 2


def test_pop() -> None:
    d = {-1: 2, 12: 24}
    assert d.pop(-1) == 2
    assert d.pop(7, 8) == 8
    assert d.pop(12, 9) == 24
    assert len(d) == 0


def test_all() -> None:
    test_dict_get()
    test_dict_del()
    test_mixed_numeric_keys()
    test_negative_keys()
    test_pop()


test_all()
