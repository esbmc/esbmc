class A:
    def __init__(self):
        self.b = 1


class B:
    def __init__(self):
        self.a = A()


o = B()
x = o.a.no_such_attr_zzz
