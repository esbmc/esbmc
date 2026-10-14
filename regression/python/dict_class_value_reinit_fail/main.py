class A:
    def __init__(self, name):
        self.aname = name


class B:
    def __init__(self, name):
        self.bname = name


def test():
    d = {}
    d[1] = A("a")
    d = {}
    d[1] = B("b")
    assert d[1].bname == "wrong"


test()
