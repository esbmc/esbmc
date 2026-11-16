class C:

    def __init__(self, a):
        self.a = a


try:
    C(a=1, b=2)
    assert False
except TypeError:
    pass
