class Klass:
    def __init__(self, name):
        self.name = name


def test_complex_keys():
    t = (1, 2, 3)
    v = (1,)
    w = (1, 2, 3)

    e = {}

    e[t] = 1
    e[v] = 2
    e[w] = 3

    assert e[t] == 3
    assert e[v] == 2
    assert e[w] == 3

    assert e == {(1, 2, 3): 3, (1,): 2}


def test_instance_value():
    d = {}
    key = 'cicero'
    d[key] = Klass(key)
    assert d[key].name == 'cicero'


def test_all():
    test_complex_keys()
    test_instance_value()


test_all()
