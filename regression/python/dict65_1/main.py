def test_dict_get():
    assert {"wah": 2}.get("aap", 3) == 3


def test_dict_del():
    d = {1: 4, 2: 5}
    del d[1]
    assert d == {2: 5}


def test_misc():
    a = {}
    a[1.0] = 1
    assert a[1.0] == 1

    b = {}
    b[1] = 1.0
    assert b[1] == 1.0

    c = {}
    c[4] = 1.0
    assert c[4] == 1.0

    d = {}
    d[4] = 1.0
    assert 4 in d


def test_all():
    test_dict_get()
    test_dict_del()
    test_misc()


test_all()
